from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_multidepth import (
    CONTEXT_FAMILIES,
    DEFAULT_NUMERIC_CONTENTS,
    DEFAULT_RELATIONS,
    MultiDepthBracketExample,
    MultiDepthResamplingSpec,
    build_active_tail,
    close_count_from_depth,
    relation_for_pair,
)
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot


SIGNATURE_VARIANTS: tuple[str, ...] = (
    "A_R_output",
    "B_D_linear",
    "B_D_sameR_linear",
    "B_D_thresholds",
    "B_D_same_surface_context",
    "C_D_frozen_readout",
)
READOUT_TARGET_NAMES: tuple[str, ...] = ("norm_D", "D_ge_2", "D_ge_3", "D_ge_4")


@dataclass(frozen=True)
class CandidateUniverse:
    csv_path: str
    csv_sha256: str
    expected_node_count: int
    rows: tuple[dict[str, str], ...]
    sites: tuple[ChannelSite, ...]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(site.site_id for site in self.sites)

    def manifest(self) -> dict[str, Any]:
        return {
            "candidate_source": "full_localized_openai_bracket_csv",
            "candidate_csv_path": self.csv_path,
            "candidate_csv_sha256": self.csv_sha256,
            "expected_node_count": int(self.expected_node_count),
            "candidate_count": len(self.sites),
            "all_candidate_node_ids": list(self.node_ids),
            "no_filtering_applied": True,
            "filtering_policy": (
                "Loaded every CSV row as a singleton neural site. No filtering by label, "
                "importance, layer, module, hook family, final_resid, or R_late."
            ),
        }


@dataclass(frozen=True)
class ReadoutQuality:
    split: str
    n: int
    threshold_macro_accuracy: float
    threshold_accuracies: dict[str, float]
    norm_depth_mae: float


@dataclass(frozen=True)
class FrozenDepthReadout:
    weights: torch.Tensor
    alpha: float
    feature_site_ids: tuple[str, ...]
    target_names: tuple[str, ...] = READOUT_TARGET_NAMES

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.to(dtype=torch.float32) @ self.weights.to(dtype=torch.float32)

    def predict(self, features: Sequence[float]) -> tuple[float, ...]:
        x = torch.tensor([list(features)], dtype=torch.float32)
        return tuple(float(v) for v in self.predict_tensor(x)[0].tolist())

    def to_json(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "feature_site_ids": list(self.feature_site_ids),
            "target_names": list(self.target_names),
            "weights": self.weights.detach().cpu().tolist(),
        }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_localized_candidate_universe(
    csv_path: str | Path,
    *,
    expected_node_count: int = 133,
) -> CandidateUniverse:
    path = Path(csv_path)
    rows = tuple(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != int(expected_node_count):
        raise ValueError(f"expected {expected_node_count} localized bracket nodes, got {len(rows)} from {path}")
    node_ids = [str(row["node_id"]) for row in rows]
    duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate node IDs in candidate CSV: {duplicates[:5]}")
    sites = tuple(
        ChannelSite.from_node_id(str(row["node_id"]), label=str(row.get("published_label") or row.get("source_key") or ""))
        for row in rows
    )
    return CandidateUniverse(
        csv_path=str(path),
        csv_sha256=file_sha256(path),
        expected_node_count=int(expected_node_count),
        rows=tuple(dict(row) for row in rows),
        sites=sites,
    )


def split_for_content_index(index: int) -> str:
    if index < 8:
        return "Dfit"
    if index < 12:
        return "Dcal"
    if index < 16:
        return "Dte"
    raise ValueError("the rich D experiment expects exactly the first 16 deterministic contents")


def encode_prompt(enc: Any | None, prompt: str) -> tuple[int, ...]:
    if enc is None:
        return tuple(ord(ch) for ch in prompt)
    return tuple(int(tok) for tok in enc.encode(prompt))


def context_prefix(context_family: str, *, depth: int, max_depth: int, content_index: int) -> str:
    if context_family == "no_distractor":
        return ""
    if context_family == "closed_pair_before":
        return f"scratch_{content_index} = [[0], [1]]\n"
    if context_family == "outside_active":
        return f"outer_{content_index} = [9]\nmeta_{content_index} = [[2]]\n"
    if context_family == "surface_balanced":
        extra = max(0, int(max_depth) - int(depth))
        return "".join(f"pad_{content_index}_{j} = [{j}]\n" for j in range(extra))
    raise ValueError(f"unknown context family: {context_family}")


def generate_content_split_multidepth_examples(
    enc: Any | None,
    *,
    depths: Sequence[int],
    numeric_contents: Sequence[str] = DEFAULT_NUMERIC_CONTENTS,
    context_families: Sequence[str] = CONTEXT_FAMILIES,
) -> tuple[MultiDepthBracketExample, ...]:
    depths = tuple(int(depth) for depth in depths)
    if not depths or min(depths) < 1:
        raise ValueError("depths must be nonempty positive integers")
    contents = tuple(str(x) for x in numeric_contents[:16])
    if len(contents) != 16:
        raise ValueError("need exactly 16 deterministic numeric contents")
    context_families = tuple(str(x) for x in context_families)
    if set(context_families) != set(CONTEXT_FAMILIES):
        raise ValueError("rich D splits must include all default context families")

    max_depth = max(depths)
    examples: list[MultiDepthBracketExample] = []
    for content_index, content in enumerate(contents):
        split = split_for_content_index(content_index)
        for depth in depths:
            for context_family in context_families:
                prefix = context_prefix(context_family, depth=depth, max_depth=max_depth, content_index=content_index)
                tail = build_active_tail(depth, content)
                prompt = prefix + tail
                token_ids = encode_prompt(enc, prompt)
                examples.append(
                    MultiDepthBracketExample(
                        example_id=f"{split}-c{content_index:02d}-d{depth}-{context_family}",
                        prompt=prompt,
                        token_ids=token_ids,
                        tail=tail,
                        depth=int(depth),
                        close_count=close_count_from_depth(depth),
                        split=split,
                        pair_id=f"content-{content_index:02d}-{context_family}",
                        context_family=context_family,
                        numeric_content=content,
                        surface_open_count=prompt.count("["),
                        surface_close_count=prompt.count("]"),
                    )
                )
    return tuple(examples)


def split_summary(examples: Sequence[MultiDepthBracketExample]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_split: dict[str, list[MultiDepthBracketExample]] = defaultdict(list)
    for ex in examples:
        by_split[ex.split].append(ex)
    for split, rows in sorted(by_split.items()):
        out[split] = {
            "n": len(rows),
            "depths": sorted({ex.depth for ex in rows}),
            "context_families": sorted({ex.context_family for ex in rows}),
            "content_indices": sorted({int(re.search(r"content-(\d+)-", ex.pair_id).group(1)) for ex in rows}),
        }
    return out


def build_relation_specs_for_split(
    examples: Sequence[MultiDepthBracketExample],
    *,
    split: str,
    max_records_per_relation: int,
    relations: Sequence[str] = DEFAULT_RELATIONS,
) -> tuple[MultiDepthResamplingSpec, ...]:
    split_examples = [ex for ex in examples if ex.split == split]
    by_relation: dict[str, list[MultiDepthResamplingSpec]] = {relation: [] for relation in relations}
    for relation in relations:
        for base in split_examples:
            for source in split_examples:
                spec = relation_for_pair(base, source, relation)
                if spec is not None:
                    by_relation[relation].append(spec)
    selected: list[MultiDepthResamplingSpec] = []
    for relation in relations:
        selected.extend(by_relation[relation][: int(max_records_per_relation)])
    return tuple(selected)


def relation_counts(specs: Sequence[MultiDepthResamplingSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec.relation] += 1
    return dict(sorted(counts.items()))


def norm_depth(depth: int | float, *, max_depth: int) -> float:
    denom = max(1, int(max_depth) - 1)
    return (float(depth) - 1.0) / float(denom)


def r_phi(example: MultiDepthBracketExample) -> float:
    return 1.0 if int(example.close_count) == 2 else -1.0


def depth_targets_for_example(example: MultiDepthBracketExample, *, max_depth: int) -> tuple[float, float, float, float]:
    d = int(example.depth)
    return (
        norm_depth(d, max_depth=max_depth),
        1.0 if d >= 2 else 0.0,
        1.0 if d >= 3 else 0.0,
        1.0 if d >= 4 else 0.0,
    )


def depth_target_matrix(examples: Sequence[MultiDepthBracketExample], *, max_depth: int) -> torch.Tensor:
    return torch.tensor([depth_targets_for_example(ex, max_depth=max_depth) for ex in examples], dtype=torch.float32)


def threshold_predictions(phi: Sequence[float]) -> tuple[int, int, int]:
    return tuple(1 if float(v) >= 0.5 else 0 for v in phi[1:4])


def threshold_targets(example: MultiDepthBracketExample) -> tuple[int, int, int]:
    d = int(example.depth)
    return (1 if d >= 2 else 0, 1 if d >= 3 else 0, 1 if d >= 4 else 0)


def abstract_components_for_spec(
    variant: str,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
    *,
    max_depth: int,
) -> tuple[float, ...]:
    base = examples[spec.base_id]
    source = examples[spec.source_id]
    d_delta = norm_depth(source.depth, max_depth=max_depth) - norm_depth(base.depth, max_depth=max_depth)
    same_r = base.close_count == source.close_count
    if variant == "A_R_output":
        return (r_phi(source) - r_phi(base),)
    if variant == "B_D_linear":
        return (d_delta,)
    if variant == "B_D_sameR_linear":
        return (d_delta if same_r else 0.0,)
    if variant == "B_D_thresholds":
        return tuple(
            float((1 if source.depth >= threshold else 0) - (1 if base.depth >= threshold else 0))
            for threshold in (2, 3, 4)
        )
    if variant == "B_D_same_surface_context":
        return (d_delta if spec.relation == "same_surface_different_active_context" else 0.0,)
    if variant == "C_D_frozen_readout":
        source_targets = depth_targets_for_example(source, max_depth=max_depth)
        base_targets = depth_targets_for_example(base, max_depth=max_depth)
        return tuple(float(s - b) for s, b in zip(source_targets, base_targets))
    raise ValueError(f"unknown signature variant: {variant}")


def abstract_signature(
    variant: str,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    *,
    max_depth: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for spec in specs:
        values.extend(abstract_components_for_spec(variant, spec, examples, max_depth=max_depth))
    return tuple(values)


def feature_names_for_signature(variant: str, specs: Sequence[MultiDepthResamplingSpec]) -> tuple[str, ...]:
    if variant == "B_D_thresholds":
        suffixes = ("D_ge_2", "D_ge_3", "D_ge_4")
    elif variant == "C_D_frozen_readout":
        suffixes = READOUT_TARGET_NAMES
    else:
        suffixes = ("scalar",)
    names: list[str] = []
    for spec in specs:
        prefix = f"{spec.relation}:{spec.base_id}->{spec.source_id}"
        names.extend(f"{prefix}:{suffix}" for suffix in suffixes)
    return tuple(names)


def clean_activation_signature_for_site(
    variant: str,
    site_id: str,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_features_by_example: Mapping[str, Mapping[str, float]],
) -> tuple[float, ...]:
    values: list[float] = []
    for spec in specs:
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        raw_delta = float(clean_features_by_example[source.example_id][site_id]) - float(
            clean_features_by_example[base.example_id][site_id]
        )
        if variant == "B_D_linear":
            values.append(raw_delta)
        elif variant == "B_D_sameR_linear":
            values.append(raw_delta if base.close_count == source.close_count else 0.0)
        elif variant == "B_D_thresholds":
            values.extend((raw_delta, raw_delta, raw_delta))
        elif variant == "B_D_same_surface_context":
            values.append(raw_delta if spec.relation == "same_surface_different_active_context" else 0.0)
        else:
            raise ValueError(f"clean activation signatures only support B variants, got {variant}")
    return tuple(values)


def c_signature_from_readout_outputs(base_phi: Sequence[float], patched_phi: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(p) - float(b) for p, b in zip(patched_phi, base_phi))


def selector_payload_from_signatures(
    *,
    variant: str,
    abstract: Sequence[float],
    neural_by_site: Mapping[str, Sequence[float]],
    epsilon: float,
    beta: float,
    cost_mode: str = "cosine",
) -> dict[str, Any]:
    site_ids = tuple(neural_by_site)
    if not site_ids:
        raise ValueError("no neural signatures")
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural_by_site[site_id]) for site_id in site_ids], dtype=torch.float32)

    squared_cost = cost_matrix(abstract_tensor, neural_tensor, mode="squared")
    squared_uot = sinkhorn_one_sided_uot(squared_cost, epsilon=float(epsilon), beta_neural=float(beta), n_iter=300)[0]

    main_cost = cost_matrix(abstract_tensor, neural_tensor, mode=cost_mode)  # type: ignore[arg-type]
    main_uot = sinkhorn_one_sided_uot(main_cost, epsilon=float(epsilon), beta_neural=float(beta), n_iter=300)[0]
    similarity = 1.0 - main_cost[0]
    direct = similarity.clamp_min(0.0)
    if float(direct.sum()) <= 0.0:
        direct = torch.softmax(similarity, dim=0)
    else:
        direct = direct / direct.sum().clamp_min(1e-12)

    selector_specs = {
        "raw_squared_uot": {
            "cost_mode": "squared",
            "coupling_rule": "one_sided_uot",
            "weights": squared_uot,
            "cost": squared_cost[0],
            "similarity": None,
        },
        "raw_cosine_uot": {
            "cost_mode": cost_mode,
            "coupling_rule": "one_sided_uot",
            "weights": main_uot,
            "cost": main_cost[0],
            "similarity": similarity,
        },
        "raw_cosine_similarity": {
            "cost_mode": cost_mode,
            "coupling_rule": "row_normalized_positive_cosine_similarity",
            "weights": direct,
            "cost": main_cost[0],
            "similarity": similarity,
        },
    }
    selectors: dict[str, Any] = {}
    for name, spec in selector_specs.items():
        rows = []
        for idx, site_id in enumerate(site_ids):
            rows.append(
                {
                    "site_id": site_id,
                    "weight": float(spec["weights"][idx]),
                    "cost": float(spec["cost"][idx]),
                    "similarity": None if spec["similarity"] is None else float(spec["similarity"][idx]),
                    "raw_signature": [float(x) for x in neural_by_site[site_id]],
                }
            )
        selectors[name] = {
            "cost_mode": spec["cost_mode"],
            "coupling_rule": spec["coupling_rule"],
            "ranked_sites": sorted(rows, key=lambda row: (-float(row["weight"]), float(row["cost"]), row["site_id"])),
        }
    return {
        "variant": variant,
        "abstract_signature": [float(x) for x in abstract],
        "site_ids": list(site_ids),
        "selectors": selectors,
        "main_selector": "raw_cosine_uot",
        "note": "Variable-specific PLOT signature. Candidate sites are the full localized bracket CSV universe.",
    }


def weights_from_ranked(ranked: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, float]:
    chosen = list(ranked[: max(1, int(k))])
    total = sum(float(row.get("weight", 0.0)) for row in chosen)
    if total <= 0.0:
        return {str(row["site_id"]): 1.0 / len(chosen) for row in chosen}
    return {str(row["site_id"]): float(row["weight"]) / total for row in chosen}


def fit_ridge_readout(x: torch.Tensor, y: torch.Tensor, *, alpha: float, feature_site_ids: Sequence[str]) -> FrozenDepthReadout:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    eye = torch.eye(x.shape[1], dtype=torch.float32, device=x.device)
    weights = torch.linalg.solve(x.T @ x + float(alpha) * eye, x.T @ y)
    return FrozenDepthReadout(
        weights=weights.detach().cpu(),
        alpha=float(alpha),
        feature_site_ids=tuple(str(site_id) for site_id in feature_site_ids),
    )


def readout_quality(readout: FrozenDepthReadout, *, split: str, x: torch.Tensor, y: torch.Tensor) -> ReadoutQuality:
    pred = readout.predict_tensor(x).detach().cpu()
    target = y.detach().cpu()
    threshold_accs: dict[str, float] = {}
    for idx, name in enumerate(("D_ge_2", "D_ge_3", "D_ge_4"), start=1):
        pred_bits = pred[:, idx] >= 0.5
        target_bits = target[:, idx] >= 0.5
        threshold_accs[name] = float((pred_bits == target_bits).float().mean().item()) if len(pred_bits) else float("nan")
    macro = float(sum(threshold_accs.values()) / len(threshold_accs)) if threshold_accs else float("nan")
    mae = float(torch.mean(torch.abs(pred[:, 0] - target[:, 0])).item()) if pred.numel() else float("nan")
    return ReadoutQuality(split=split, n=int(x.shape[0]), threshold_macro_accuracy=macro, threshold_accuracies=threshold_accs, norm_depth_mae=mae)


def is_readout_valid(qualities: Mapping[str, ReadoutQuality], *, threshold: float = 0.90) -> bool:
    return (
        "Dcal" in qualities
        and "Dte" in qualities
        and qualities["Dcal"].threshold_macro_accuracy >= threshold
        and qualities["Dte"].threshold_macro_accuracy >= threshold
    )


def vector_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def readout_moves_toward_source(*, base_phi: Sequence[float], source_phi: Sequence[float], patched_phi: Sequence[float]) -> bool:
    base_to_source = vector_distance(base_phi, source_phi)
    if base_to_source <= 1e-6:
        return False
    patched_to_source = vector_distance(patched_phi, source_phi)
    return patched_to_source + 1e-6 < base_to_source


def readout_preserves_base(*, base_example: MultiDepthBracketExample, patched_phi: Sequence[float]) -> bool:
    return threshold_predictions(patched_phi) == threshold_targets(base_example)


def summarize_validation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)

    def rows(name: str) -> list[Mapping[str, Any]]:
        return by_relation.get(name, [])

    def mean(values: Iterable[Any]) -> float:
        vals = [float(x) for x in values]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "records": len(records),
        "relation_counts": {key: len(val) for key, val in sorted(by_relation.items())},
        "same_D_output_preserve": mean(row["output_preserves_base"] for row in rows("same_D")),
        "different_D_same_R_readout_move": mean(row["readout_moves_toward_source"] for row in rows("different_D_same_R")),
        "different_D_same_R_output_flip": mean(row["output_flips"] for row in rows("different_D_same_R")),
        "different_D_different_R_readout_move": mean(row["readout_moves_toward_source"] for row in rows("different_D_different_R")),
        "different_D_different_R_output_flip_to_source": mean(row["output_matches_source"] for row in rows("different_D_different_R")),
        "same_surface_readout_move": mean(row["readout_moves_toward_source"] for row in rows("same_surface_different_active_context")),
        "same_surface_output_preserve": mean(row["output_preserves_base"] for row in rows("same_surface_different_active_context")),
        "wrong_numeric_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_numeric_content")),
        "wrong_numeric_readout_preserve": mean(row["readout_preserves_base"] for row in rows("wrong_numeric_content")),
        "wrong_tail_length_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_tail_length")),
        "wrong_tail_length_readout_preserve": mean(row["readout_preserves_base"] for row in rows("wrong_tail_length")),
    }


def d_acceptance(summary: Mapping[str, Any], *, threshold: float = 0.90) -> dict[str, Any]:
    checks = {
        "different_D_same_R_readout_move": float(summary.get("different_D_same_R_readout_move", float("nan"))) >= threshold,
        "different_D_same_R_output_flip_le_0.10": float(summary.get("different_D_same_R_output_flip", float("nan"))) <= 0.10,
        "different_D_different_R_readout_move": float(summary.get("different_D_different_R_readout_move", float("nan"))) >= threshold,
        "different_D_different_R_output_flip_to_source": float(summary.get("different_D_different_R_output_flip_to_source", float("nan"))) >= threshold,
        "same_D_output_preserve": float(summary.get("same_D_output_preserve", float("nan"))) >= threshold,
        "wrong_numeric_output_preserve": float(summary.get("wrong_numeric_output_preserve", float("nan"))) >= threshold,
        "wrong_numeric_readout_preserve": float(summary.get("wrong_numeric_readout_preserve", float("nan"))) >= threshold,
        "wrong_tail_length_output_preserve": float(summary.get("wrong_tail_length_output_preserve", float("nan"))) >= threshold,
        "wrong_tail_length_readout_preserve": float(summary.get("wrong_tail_length_readout_preserve", float("nan"))) >= threshold,
    }
    return {"D_validated": all(checks.values()), "checks": checks}


def d_calibration_score(summary: Mapping[str, Any]) -> float:
    values = [
        float(summary.get("different_D_same_R_readout_move", 0.0)),
        1.0 - float(summary.get("different_D_same_R_output_flip", 1.0)),
        float(summary.get("different_D_different_R_readout_move", 0.0)),
        float(summary.get("different_D_different_R_output_flip_to_source", 0.0)),
        float(summary.get("same_D_output_preserve", 0.0)),
        float(summary.get("wrong_numeric_output_preserve", 0.0)),
        float(summary.get("wrong_numeric_readout_preserve", 0.0)),
        float(summary.get("wrong_tail_length_output_preserve", 0.0)),
        float(summary.get("wrong_tail_length_readout_preserve", 0.0)),
    ]
    return float(sum(values) / len(values))


def r_calibration_score(records: Sequence[Mapping[str, Any]]) -> float:
    same = [row for row in records if row["base_close_count"] == row["source_close_count"]]
    different = [row for row in records if row["base_close_count"] != row["source_close_count"]]
    wrong = [row for row in records if row.get("wrong_variable")]

    def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(sum(float(row[key]) for row in rows) / len(rows)) if rows else 0.0

    return (mean(same, "output_preserves_base") + mean(different, "output_matches_source") + mean(wrong, "output_preserves_base")) / 3.0


def selected_support_overlap(site_ids: Sequence[str]) -> dict[str, Any]:
    r_late = {"7.mlp.post_act:4133", "7.mlp.resid_delta:2041"}
    site_set = set(str(x) for x in site_ids)
    final_resid = sorted(site_id for site_id in site_set if site_id.startswith("final_resid:"))
    r_overlap = sorted(site_set & r_late)
    return {
        "r_late_overlap": r_overlap,
        "final_resid_overlap": final_resid,
        "late_or_readout_overlapping": bool(r_overlap or final_resid),
    }


def quality_to_json(row: ReadoutQuality) -> dict[str, Any]:
    return {
        "split": row.split,
        "n": row.n,
        "threshold_macro_accuracy": row.threshold_macro_accuracy,
        "threshold_accuracies": row.threshold_accuracies,
        "norm_depth_mae": row.norm_depth_mae,
    }
