from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .activation import ChannelSite, find_first_quote_token_position
from .bracket_progressive_model_discovery import (
    DISCOVERY_RELATIONS,
    MultiDepthBracketExample,
    build_relation_specs_for_split,
)
from .bracket_group_depth_plot import build_unique_discovery_bank
from .handle_necessity import QUOTE_TEMPLATES, load_candidate_circuit, quote_contents
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .runtime import quote_token_ids


@dataclass(frozen=True)
class RediscoveryExample:
    example_id: str
    prompt: str
    token_ids: tuple[int, ...]
    split: str
    content_id: str
    variable_value: int
    patch_position: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RediscoveryPair:
    relation: str
    base_id: str
    source_id: str


@dataclass(frozen=True)
class ClampedRun:
    example_id: str
    class_margin: float
    predicted_value: int
    features_by_site: Mapping[str, float]


@dataclass(frozen=True)
class HandleConfiguration:
    handle_id: str
    weights_by_site: Mapping[str, float]
    strength: float


def split_for_index(index: int, *, fit: int, cal: int, test: int) -> str:
    if int(index) < int(fit):
        return "Dfit"
    if int(index) < int(fit) + int(cal):
        return "Dcal"
    if int(index) < int(fit) + int(cal) + int(test):
        return "Dte"
    raise ValueError("content index outside declared splits")


def build_quote_rediscovery_bank(
    enc: Any,
    *,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
    content_offset: int,
) -> tuple[RediscoveryExample, ...]:
    count = int(fit_contents) + int(cal_contents) + int(test_contents)
    contents = quote_contents(count, offset=int(content_offset))
    examples: list[RediscoveryExample] = []
    for index, content in enumerate(contents):
        split = split_for_index(index, fit=fit_contents, cal=cal_contents, test=test_contents)
        content_id = f"quote-{int(content_offset) + index:06d}"
        for template_id, template in QUOTE_TEMPLATES:
            for quote_type, quote, value in (("single", "'", -1), ("double", '"', 1)):
                prompt = template.format(quote=quote, content=content)
                token_ids = tuple(int(x) for x in enc.encode(prompt))
                opener = find_first_quote_token_position(enc, token_ids, quote)
                examples.append(
                    RediscoveryExample(
                        example_id=f"{split}-{content_id}-{template_id}-{quote_type}",
                        prompt=prompt,
                        token_ids=token_ids,
                        split=split,
                        content_id=content_id,
                        variable_value=int(value),
                        patch_position=int(opener),
                        metadata={
                            "task": "quote",
                            "template": template_id,
                            "quote_type": quote_type,
                            "content": content,
                            "pair_id": f"{content_id}-{template_id}",
                        },
                    )
                )
    return tuple(examples)


def build_bracket_rediscovery_bank(
    enc: Any,
    *,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
    content_offset: int,
) -> tuple[RediscoveryExample, ...]:
    rows = build_unique_discovery_bank(
        enc,
        contents=int(fit_contents) + int(cal_contents) + int(test_contents),
        fit_contents=int(fit_contents),
        cal_contents=int(cal_contents),
        test_contents=int(test_contents),
        content_offset=int(content_offset),
        depths=(1, 2, 3, 4),
    )
    return tuple(
        RediscoveryExample(
            example_id=row.example_id,
            prompt=row.prompt,
            token_ids=tuple(int(x) for x in row.token_ids),
            split=row.split,
            content_id=str(row.numeric_content),
            variable_value=1 if int(row.close_count) == 2 else -1,
            patch_position=len(row.token_ids) - 1,
            metadata={
                "task": "bracket",
                "depth": int(row.depth),
                "close_count": int(row.close_count),
                "context_family": row.context_family,
                "numeric_content": row.numeric_content,
                "tail": row.tail,
                "pair_id": row.pair_id,
                "surface_open_count": int(row.surface_open_count),
                "surface_close_count": int(row.surface_close_count),
            },
        )
        for row in rows
    )


def _balanced_prefix(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    limit: int,
) -> tuple[RediscoveryPair, ...]:
    if len(pairs) <= int(limit):
        return tuple(pairs)
    by_base_value: dict[int, list[RediscoveryPair]] = defaultdict(list)
    for pair in pairs:
        by_base_value[int(examples[pair.base_id].variable_value)].append(pair)
    output: list[RediscoveryPair] = []
    values = sorted(by_base_value)
    while len(output) < int(limit) and any(by_base_value.values()):
        for value in values:
            if by_base_value[value] and len(output) < int(limit):
                output.append(by_base_value[value].pop(0))
    return tuple(output)


def build_quote_pairs(
    examples: Sequence[RediscoveryExample],
    *,
    split: str,
    records_per_relation: int,
) -> tuple[RediscoveryPair, ...]:
    rows = sorted((row for row in examples if row.split == split), key=lambda row: row.example_id)
    by_id = {row.example_id: row for row in rows}
    selected: list[RediscoveryPair] = []
    for relation in ("different_variable", "same_variable"):
        candidates: list[RediscoveryPair] = []
        for base in rows:
            for source in rows:
                if base.example_id == source.example_id:
                    continue
                same_value = int(base.variable_value) == int(source.variable_value)
                same_pair = str(base.metadata["pair_id"]) == str(source.metadata["pair_id"])
                if relation == "different_variable" and not same_value and same_pair:
                    candidates.append(RediscoveryPair(relation, base.example_id, source.example_id))
                elif relation == "same_variable" and same_value and not same_pair:
                    candidates.append(RediscoveryPair(relation, base.example_id, source.example_id))
        selected.extend(_balanced_prefix(candidates, by_id, int(records_per_relation)))
    return tuple(selected)


def _as_multidepth(examples: Sequence[RediscoveryExample]) -> tuple[MultiDepthBracketExample, ...]:
    return tuple(
        MultiDepthBracketExample(
            example_id=row.example_id,
            prompt=row.prompt,
            token_ids=row.token_ids,
            tail=str(row.metadata["tail"]),
            depth=int(row.metadata["depth"]),
            close_count=int(row.metadata["close_count"]),
            split=row.split,
            pair_id=str(row.metadata["pair_id"]),
            context_family=str(row.metadata["context_family"]),
            numeric_content=str(row.metadata["numeric_content"]),
            surface_open_count=int(row.metadata["surface_open_count"]),
            surface_close_count=int(row.metadata["surface_close_count"]),
        )
        for row in examples
    )


def build_bracket_pairs(
    examples: Sequence[RediscoveryExample],
    *,
    split: str,
    records_per_relation: int,
) -> tuple[RediscoveryPair, ...]:
    specs = build_relation_specs_for_split(
        _as_multidepth(examples),
        split=split,
        records_per_relation=int(records_per_relation),
        relations=DISCOVERY_RELATIONS,
    )
    return tuple(RediscoveryPair(row.relation, row.base_id, row.source_id) for row in specs)


def bank_manifest(
    examples: Sequence[RediscoveryExample],
    pairs_by_split: Mapping[str, Sequence[RediscoveryPair]],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    content_sets: list[set[str]] = []
    for split in ("Dfit", "Dcal", "Dte"):
        rows = [row for row in examples if row.split == split]
        contents = {row.content_id for row in rows}
        content_sets.append(contents)
        counts: dict[str, int] = defaultdict(int)
        for pair in pairs_by_split[split]:
            counts[pair.relation] += 1
        splits[split] = {
            "examples": len(rows),
            "content_count": len(contents),
            "content_ids": sorted(contents),
            "pair_counts": dict(sorted(counts.items())),
        }
    return {
        "splits": splits,
        "content_splits_disjoint": all(
            not (left & right)
            for index, left in enumerate(content_sets)
            for right in content_sets[index + 1 :]
        ),
    }


def abstract_signature(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
) -> tuple[float, ...]:
    return tuple(
        float(examples[pair.source_id].variable_value - examples[pair.base_id].variable_value)
        for pair in pairs
    )


def _hook_regex(sites: Sequence[ChannelSite]) -> str:
    hooks = sorted({site.hook_key for site in sites})
    return "^(?:" + "|".join(re.escape(value) for value in hooks) + ")$"


def _clamp_assignments(
    disabled_sites: Sequence[ChannelSite],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for site in disabled_sites:
        grouped[site.hook_key].append(int(site.channel))
    return {key: tuple(values) for key, values in grouped.items()}


def _class_margin(
    logits: torch.Tensor,
    *,
    negative_token_id: int,
    positive_token_id: int,
) -> torch.Tensor:
    last = logits[:, -1, :].to(torch.float32)
    return last[:, int(positive_token_id)] - last[:, int(negative_token_id)]


def collect_clamped_runs(
    model: Any,
    examples: Sequence[RediscoveryExample],
    *,
    candidate_sites: Sequence[ChannelSite],
    disabled_sites: Sequence[ChannelSite],
    hook_means: Mapping[str, torch.Tensor],
    negative_token_id: int,
    positive_token_id: int,
    device: str,
    max_batch_size: int,
) -> dict[str, ClampedRun]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    clamped = _clamp_assignments(disabled_sites)
    sites_by_hook: dict[str, list[ChannelSite]] = defaultdict(list)
    for site in candidate_sites:
        sites_by_hook[site.hook_key].append(site)
    by_length: dict[int, list[RediscoveryExample]] = defaultdict(list)
    for row in examples:
        by_length[len(row.token_ids)].append(row)
    output: dict[str, ClampedRun] = {}
    for rows in by_length.values():
        for start in range(0, len(rows), int(max_batch_size)):
            batch = rows[start : start + int(max_batch_size)]
            token_ids = torch.tensor([row.token_ids for row in batch], dtype=torch.long, device=device)
            positions = torch.tensor([row.patch_position for row in batch], dtype=torch.long, device=device)
            batch_rows = torch.arange(len(batch), dtype=torch.long, device=device)
            recorded: dict[str, torch.Tensor] = {}
            interventions: dict[str, Any] = {}
            for hook_key in sorted(sites_by_hook):
                channels = clamped.get(hook_key, ())
                hook_sites = tuple(sites_by_hook[hook_key])
                mean_cpu = hook_means[hook_key]

                def _clamp_and_record(
                    tensor: torch.Tensor,
                    *,
                    channels: tuple[int, ...] = channels,
                    hook_sites: tuple[ChannelSite, ...] = hook_sites,
                    mean_cpu: torch.Tensor = mean_cpu,
                ) -> torch.Tensor:
                    if channels:
                        patched = tensor.clone()
                        mean = mean_cpu.to(device=tensor.device, dtype=tensor.dtype)
                        for channel in channels:
                            patched[..., int(channel)] = mean[int(channel)]
                    else:
                        patched = tensor
                    for site in hook_sites:
                        recorded[site.site_id] = patched[
                            batch_rows, positions, int(site.channel)
                        ].detach().cpu()
                    return patched

                interventions[hook_key] = _clamp_and_record
            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    logits, _, _ = model(token_ids)
            margins = _class_margin(
                logits,
                negative_token_id=negative_token_id,
                positive_token_id=positive_token_id,
            ).detach().cpu()
            missing = {site.site_id for site in candidate_sites} - set(recorded)
            if missing:
                raise RuntimeError(f"clean run did not record candidate sites: {sorted(missing)[:5]}")
            for batch_index, row in enumerate(batch):
                features = {
                    site.site_id: float(recorded[site.site_id][batch_index])
                    for site in candidate_sites
                }
                margin = float(margins[batch_index])
                output[row.example_id] = ClampedRun(
                    example_id=row.example_id,
                    class_margin=margin,
                    predicted_value=1 if margin > 0.0 else -1,
                    features_by_site=features,
                )
    return output


def evaluate_configurations(
    model: Any,
    configs: Sequence[HandleConfiguration],
    pairs: Sequence[RediscoveryPair],
    *,
    examples: Mapping[str, RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    site_lookup: Mapping[str, ChannelSite],
    disabled_sites: Sequence[ChannelSite],
    hook_means: Mapping[str, torch.Tensor],
    negative_token_id: int,
    positive_token_id: int,
    device: str,
    max_batch_size: int,
) -> np.ndarray:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not configs or not pairs:
        raise ValueError("configs and pairs must be nonempty")
    clamped = _clamp_assignments(disabled_sites)
    result = np.empty((len(configs), len(pairs)), dtype=np.float32)
    pairs_by_length: dict[int, list[int]] = defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        pairs_by_length[len(examples[pair.base_id].token_ids)].append(pair_index)
    for pair_indices in pairs_by_length.values():
        task_count = len(configs) * len(pair_indices)
        for task_start in range(0, task_count, int(max_batch_size)):
            flat = list(range(task_start, min(task_count, task_start + int(max_batch_size))))
            config_indices = [value // len(pair_indices) for value in flat]
            local_pair_indices = [value % len(pair_indices) for value in flat]
            global_pair_indices = [pair_indices[value] for value in local_pair_indices]
            base_rows = [examples[pairs[index].base_id] for index in global_pair_indices]
            token_ids = torch.tensor([row.token_ids for row in base_rows], dtype=torch.long, device=device)
            patches_by_hook: dict[str, list[tuple[int, int, int, float, float]]] = defaultdict(list)
            for batch_row, (config_index, pair_index) in enumerate(zip(config_indices, global_pair_indices)):
                config = configs[config_index]
                pair = pairs[pair_index]
                source_run = runs[pair.source_id]
                position = int(examples[pair.base_id].patch_position)
                for site_id, weight in config.weights_by_site.items():
                    site = site_lookup[site_id]
                    patches_by_hook[site.hook_key].append(
                        (
                            batch_row,
                            position,
                            int(site.channel),
                            float(source_run.features_by_site[site_id]),
                            float(config.strength) * float(weight),
                        )
                    )
            active_hooks = sorted(set(clamped) | set(patches_by_hook))
            interventions: dict[str, Any] = {}
            for hook_key in active_hooks:
                channels = clamped.get(hook_key, ())
                patches = tuple(patches_by_hook.get(hook_key, ()))
                mean_cpu = hook_means[hook_key]

                def _intervene(
                    tensor: torch.Tensor,
                    *,
                    channels: tuple[int, ...] = channels,
                    patches: tuple[tuple[int, int, int, float, float], ...] = patches,
                    mean_cpu: torch.Tensor = mean_cpu,
                ) -> torch.Tensor:
                    patched = tensor.clone()
                    mean = mean_cpu.to(device=tensor.device, dtype=tensor.dtype)
                    for channel in channels:
                        patched[..., int(channel)] = mean[int(channel)]
                    for row, position, channel, source_value, coefficient in patches:
                        current = patched[int(row), int(position), int(channel)]
                        source = torch.as_tensor(source_value, device=tensor.device, dtype=tensor.dtype)
                        patched[int(row), int(position), int(channel)] = current + float(coefficient) * (
                            source - current
                        )
                    return patched

                interventions[hook_key] = _intervene
            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    logits, _, _ = model(token_ids)
            margins = _class_margin(
                logits,
                negative_token_id=negative_token_id,
                positive_token_id=positive_token_id,
            ).detach().cpu().numpy()
            for batch_row, (config_index, pair_index) in enumerate(zip(config_indices, global_pair_indices)):
                result[config_index, pair_index] = float(margins[batch_row])
    return result


def match_signatures(
    abstract: Sequence[float],
    neural_by_site: Mapping[str, Sequence[float]],
    *,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    site_ids = tuple(neural_by_site)
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural_by_site[site_id]) for site_id in site_ids], dtype=torch.float32)
    costs = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    coupling = sinkhorn_one_sided_uot(
        costs,
        epsilon=float(epsilon),
        beta_neural=float(beta),
        n_iter=300,
    )[0]
    ranked = sorted(
        (
            {
                "site_id": site_id,
                "weight": float(coupling[index]),
                "cost": float(costs[0, index]),
            }
            for index, site_id in enumerate(site_ids)
        ),
        key=lambda row: (-float(row["weight"]), float(row["cost"]), str(row["site_id"])),
    )
    return {
        "cost_mode": "raw_cosine",
        "matching": "one_sided_uot",
        "epsilon": float(epsilon),
        "beta": float(beta),
        "ranked": ranked,
    }


def normalized_topk_weights(ranked: Sequence[Mapping[str, Any]], k: int) -> dict[str, float]:
    rows = list(ranked[: int(k)])
    total = sum(float(row["weight"]) for row in rows)
    if total <= 0.0:
        return {str(row["site_id"]): 1.0 / len(rows) for row in rows}
    return {str(row["site_id"]): float(row["weight"]) / total for row in rows}


def relation_summary(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    patched_margins: Sequence[float],
) -> dict[str, Any]:
    by_relation: dict[str, list[bool]] = defaultdict(list)
    for pair, margin in zip(pairs, patched_margins):
        base_value = int(examples[pair.base_id].variable_value)
        source_value = int(examples[pair.source_id].variable_value)
        predicted = 1 if float(margin) > 0.0 else -1
        expected = source_value if source_value != base_value else base_value
        by_relation[pair.relation].append(predicted == expected)
    rates = {relation: float(np.mean(values)) for relation, values in sorted(by_relation.items())}
    return {
        "rates": rates,
        "score": float(np.mean(list(rates.values()))) if rates else float("nan"),
        "all_rates_at_least_0_90": bool(rates and min(rates.values()) >= 0.90),
    }


def clean_accuracy(
    examples: Sequence[RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    *,
    split: str,
) -> float:
    rows = [row for row in examples if row.split == split]
    return float(np.mean([runs[row.example_id].predicted_value == row.variable_value for row in rows]))


def select_calibration_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("empty calibration grid")
    return sorted(
        rows,
        key=lambda row: (
            -float(row["summary"]["score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]


def can_certify_redundancy(*, clean_dte_accuracy: float, heldout_summary: Mapping[str, Any]) -> bool:
    return bool(float(clean_dte_accuracy) >= 0.90 and heldout_summary["all_rates_at_least_0_90"])


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)
