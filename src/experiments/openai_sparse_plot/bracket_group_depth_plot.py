from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_d_rich_signatures import depth_targets_for_example
from .bracket_multidepth import (
    CONTEXT_FAMILIES,
    MultiDepthBracketExample,
    MultiDepthResamplingSpec,
    build_active_tail,
    close_count_from_depth,
)
from .bracket_progressive_model_discovery import context_prefix, encode_prompt


TARGET_COMPONENT_INDEX: dict[str, int | None] = {
    "D": None,
    "T2": 1,
    "T3": 2,
    "T4": 3,
}


@dataclass(frozen=True)
class GroupCandidate:
    group_id: str
    site_indices: tuple[int, ...]
    site_ids: tuple[str, ...]

    @property
    def group_size(self) -> int:
        return len(self.site_indices)


def group_id_for(site_ids: Sequence[str]) -> str:
    return " + ".join(str(site_id) for site_id in site_ids)


def build_single_pair_groups(sites: Sequence[ChannelSite]) -> tuple[GroupCandidate, ...]:
    groups: list[GroupCandidate] = []
    for index, site in enumerate(sites):
        groups.append(GroupCandidate(site.site_id, (index,), (site.site_id,)))
    for first in range(len(sites)):
        for second in range(first + 1, len(sites)):
            site_ids = (sites[first].site_id, sites[second].site_id)
            groups.append(GroupCandidate(group_id_for(site_ids), (first, second), site_ids))
    return tuple(groups)


def build_progressive_triples(
    sites: Sequence[ChannelSite],
    seed_pairs: Sequence[GroupCandidate],
) -> tuple[GroupCandidate, ...]:
    triples: dict[tuple[int, int, int], GroupCandidate] = {}
    for pair in seed_pairs:
        if pair.group_size != 2:
            raise ValueError("progressive triple seeds must all be pairs")
        pair_indices = set(pair.site_indices)
        for third in range(len(sites)):
            if third in pair_indices:
                continue
            indices = tuple(sorted((*pair.site_indices, third)))
            site_ids = tuple(sites[index].site_id for index in indices)
            triples[indices] = GroupCandidate(group_id_for(site_ids), indices, site_ids)
    return tuple(triples[key] for key in sorted(triples))


def full_swap_weights(group: GroupCandidate) -> dict[str, float]:
    """Return canonical group-site swap coefficients.

    A group is one candidate neural site. Every member coordinate is replaced
    fully by its source value when the group's Dfit signature is measured.
    Coefficients therefore equal one and are deliberately not normalized by
    group size.
    """

    return {site_id: 1.0 for site_id in group.site_ids}


def abstract_delta_matrix(
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    *,
    max_depth: int = 4,
) -> torch.Tensor:
    rows = []
    for spec in specs:
        base = depth_targets_for_example(examples[spec.base_id], max_depth=max_depth)
        source = depth_targets_for_example(examples[spec.source_id], max_depth=max_depth)
        rows.append([float(source_value - base_value) for source_value, base_value in zip(source, base)])
    return torch.tensor(rows, dtype=torch.float32)


def target_signature(matrix: torch.Tensor, target: str) -> torch.Tensor:
    if target not in TARGET_COMPONENT_INDEX:
        raise ValueError(f"unknown target: {target}")
    if matrix.ndim != 2 or int(matrix.shape[1]) != 4:
        raise ValueError("effect matrix must have shape [records, 4]")
    component = TARGET_COMPONENT_INDEX[target]
    if component is None:
        return matrix.reshape(-1)
    return matrix[:, int(component)].reshape(-1)


def cosine_scores(abstract: torch.Tensor, neural: torch.Tensor) -> torch.Tensor:
    abstract = abstract.to(dtype=torch.float32).reshape(1, -1)
    neural = neural.to(dtype=torch.float32)
    if neural.ndim != 2 or int(neural.shape[1]) != int(abstract.shape[1]):
        raise ValueError("neural signatures must have shape [groups, signature_dim]")
    abstract_norm = abstract.norm(dim=1).clamp_min(1e-12)
    neural_norm = neural.norm(dim=1).clamp_min(1e-12)
    scores = (neural @ abstract[0]) / (neural_norm * abstract_norm[0])
    scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1.0))
    return scores


def rank_groups_by_cosine(
    abstract: torch.Tensor,
    neural: torch.Tensor,
    groups: Sequence[GroupCandidate],
) -> list[dict[str, Any]]:
    if len(groups) != int(neural.shape[0]):
        raise ValueError("group count does not match neural signature rows")
    scores = cosine_scores(abstract, neural)
    rows = [
        {
            "group_index": index,
            "group_id": group.group_id,
            "site_ids": list(group.site_ids),
            "group_size": group.group_size,
            "cosine_similarity": float(scores[index]),
            "cosine_cost": float(1.0 - scores[index]),
        }
        for index, group in enumerate(groups)
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["cosine_similarity"]),
            int(row["group_size"]),
            str(row["group_id"]),
        ),
    )


def interaction_residual(
    pair_signature: torch.Tensor,
    first_signature: torch.Tensor,
    second_signature: torch.Tensor,
) -> torch.Tensor:
    if pair_signature.shape != first_signature.shape or pair_signature.shape != second_signature.shape:
        raise ValueError("interaction signatures must have identical shapes")
    return pair_signature - first_signature - second_signature


def passing_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row.get("validated", False))]


def select_pass_first(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select a calibration result without allowing an average to hide a failed gate."""

    if not rows:
        raise ValueError("cannot select from an empty calibration grid")
    passed = passing_rows(rows)
    if not passed:
        return sorted(
            rows,
            key=lambda row: (
                -float(row.get("worst_gate", -math.inf)),
                -float(row.get("calibration_score", -math.inf)),
                int(row.get("distinct_site_count", len(row.get("site_ids", ())))),
                abs(float(row.get("strength", 1.0)) - 1.0),
                int(row.get("ranking_position", 10**9)),
                str(row.get("group_id", row.get("handle_id", ""))),
            ),
        )[0]
    return sorted(
        passed,
        key=lambda row: (
            int(row.get("distinct_site_count", len(row.get("site_ids", ())))),
            -float(row.get("worst_gate", -math.inf)),
            -float(row.get("calibration_score", -math.inf)),
            abs(float(row.get("strength", 1.0)) - 1.0),
            int(row.get("ranking_position", 10**9)),
            str(row.get("group_id", row.get("handle_id", ""))),
        ),
    )[0]


def unique_numeric_contents(n: int, *, offset: int = 0) -> tuple[str, ...]:
    """Generate unique, small-integer list contents for sealed confirmation banks."""

    if int(n) <= 0:
        raise ValueError("content count must be positive")
    if int(offset) < 0:
        raise ValueError("content offset must be nonnegative")
    contents: list[str] = []
    for local_index in range(int(n)):
        value = int(offset) + local_index
        digits = []
        remaining = value
        for _ in range(4):
            digits.append(remaining % 23)
            remaining //= 23
        if remaining:
            raise ValueError("content offset exceeds the four-digit base-23 range")
        suffix = [
            (value * 7 + 3) % 23,
            (value * 11 + 5) % 23,
            (value * 13 + 9) % 23,
            (value * 17 + 1) % 23,
        ]
        contents.append(", ".join(str(number) for number in (*digits, *suffix)))
    if len(set(contents)) != len(contents):
        raise AssertionError("unique content generator produced a collision")
    return tuple(contents)


def split_name_for_local_index(
    index: int,
    *,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
) -> str:
    if index < int(fit_contents):
        return "Dfit"
    if index < int(fit_contents) + int(cal_contents):
        return "Dcal"
    if index < int(fit_contents) + int(cal_contents) + int(test_contents):
        return "Dte"
    raise ValueError("content index exceeds configured split sizes")


def build_unique_discovery_bank(
    enc: Any | None,
    *,
    contents: int,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
    content_offset: int,
    depths: Sequence[int] = (1, 2, 3, 4),
    context_families: Sequence[str] = CONTEXT_FAMILIES,
) -> tuple[MultiDepthBracketExample, ...]:
    if int(contents) != int(fit_contents) + int(cal_contents) + int(test_contents):
        raise ValueError("contents must equal fit_contents + cal_contents + test_contents")
    depths = tuple(int(depth) for depth in depths)
    if not depths or min(depths) < 1:
        raise ValueError("depths must be positive")
    if set(context_families) != set(CONTEXT_FAMILIES):
        raise ValueError("confirmation bank must include every default context family")
    numeric_contents = unique_numeric_contents(int(contents), offset=int(content_offset))
    max_depth = max(depths)
    examples: list[MultiDepthBracketExample] = []
    for local_index, content in enumerate(numeric_contents):
        global_index = int(content_offset) + local_index
        split = split_name_for_local_index(
            local_index,
            fit_contents=int(fit_contents),
            cal_contents=int(cal_contents),
            test_contents=int(test_contents),
        )
        for depth in depths:
            for context_family in context_families:
                prefix = context_prefix(
                    context_family,
                    depth=depth,
                    max_depth=max_depth,
                    content_index=global_index,
                )
                tail = build_active_tail(depth, content)
                prompt = prefix + tail
                examples.append(
                    MultiDepthBracketExample(
                        example_id=f"{split}-u{global_index:06d}-d{depth}-{context_family}",
                        prompt=prompt,
                        token_ids=encode_prompt(enc, prompt),
                        tail=tail,
                        depth=int(depth),
                        close_count=close_count_from_depth(depth),
                        split=split,
                        pair_id=f"unique-{global_index:06d}-{context_family}",
                        context_family=context_family,
                        numeric_content=content,
                        surface_open_count=prompt.count("["),
                        surface_close_count=prompt.count("]"),
                    )
                )
    return tuple(examples)
