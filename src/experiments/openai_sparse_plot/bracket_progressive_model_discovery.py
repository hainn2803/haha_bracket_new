from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_multidepth import (
    CONTEXT_FAMILIES,
    MultiDepthBracketExample,
    MultiDepthResamplingSpec,
    build_active_tail,
    close_count_from_depth,
    numbers_only,
)
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .run_bracket_d_rich_signature_experiments import (
    CleanRun,
    _bracket_margin,
    _clean_summary,
    _collect_clean_runs,
    _feature_dict,
    _feature_vector,
    _hook_regex,
    _make_weighted_patch_from_features,
    _run_weighted_patch,
    _sign_from_margin,
)


DISCOVERY_RELATIONS: tuple[str, ...] = (
    "different_R",
    "same_R_different_D",
    "same_D",
    "same_surface_different_active_context",
    "wrong_numeric_content",
    "wrong_tail_length",
)

H_RLATE_SITE_IDS: tuple[str, ...] = ("7.mlp.post_act:4133", "7.mlp.resid_delta:2041")
POSTHOC_SITE_IDS: tuple[str, ...] = (
    "4.attn.resid_delta:1079",
    "2.attn.resid_delta:1249",
    "7.mlp.post_act:4133",
    "7.mlp.resid_delta:2041",
)

DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_progressive_discovery_20260708_balanced")


@dataclass(frozen=True)
class CandidateUniverse:
    csv_path: str
    csv_sha256: str
    rows: tuple[dict[str, str], ...]
    sites: tuple[ChannelSite, ...]
    expected_candidate_count: int

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(site.site_id for site in self.sites)

    def manifest(self) -> dict[str, Any]:
        return {
            "candidate_source": "full_localized_openai_bracket_csv",
            "candidate_csv_path": self.csv_path,
            "candidate_csv_sha256": self.csv_sha256,
            "candidate_count": len(self.sites),
            "expected_candidate_count": int(self.expected_candidate_count),
            "all_candidate_node_ids": list(self.node_ids),
            "no_filtering_applied": True,
        }


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    component_names: tuple[str, ...]
    frozen_readout_site_ids: tuple[str, ...] = ()
    frozen_readout_weights: Mapping[str, float] | None = None
    frozen_mid_site_ids: tuple[str, ...] = ()
    frozen_mid_weights: Mapping[str, float] | None = None
    acceptance_eligible: bool = True


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidate_universe(csv_path: str | Path, *, expected_candidate_count: int = 133) -> CandidateUniverse:
    path = Path(csv_path)
    rows = tuple(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != int(expected_candidate_count):
        raise ValueError(f"expected {expected_candidate_count} candidates, got {len(rows)} from {path}")
    node_ids = [str(row["node_id"]) for row in rows]
    duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate candidate node IDs: {duplicates[:5]}")
    sites = tuple(
        ChannelSite.from_node_id(str(row["node_id"]), label=str(row.get("published_label") or row.get("source_key") or ""))
        for row in rows
    )
    return CandidateUniverse(
        csv_path=str(path),
        csv_sha256=file_sha256(path),
        rows=tuple(dict(row) for row in rows),
        sites=sites,
        expected_candidate_count=int(expected_candidate_count),
    )


def deterministic_numeric_contents(n: int) -> tuple[str, ...]:
    if int(n) <= 0:
        raise ValueError("content count must be positive")
    contents: list[str] = []
    for i in range(int(n)):
        length = 1 + ((i * 5 + 3) % 7)
        values = [str((i * 7 + j * 11 + 3) % 23) for j in range(length)]
        contents.append(", ".join(values))
    return tuple(contents)


def split_name_for_content_index(index: int, *, fit_contents: int, cal_contents: int, test_contents: int) -> str:
    if index < int(fit_contents):
        return "Dfit"
    if index < int(fit_contents) + int(cal_contents):
        return "Dcal"
    if index < int(fit_contents) + int(cal_contents) + int(test_contents):
        return "Dte"
    raise ValueError("content index exceeds configured split sizes")


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


def encode_prompt(enc: Any | None, prompt: str) -> tuple[int, ...]:
    if enc is None:
        return tuple(ord(ch) for ch in prompt)
    return tuple(int(tok) for tok in enc.encode(prompt))


def build_discovery_bank(
    enc: Any | None,
    *,
    contents: int,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
    depths: Sequence[int] = (1, 2, 3, 4),
    context_families: Sequence[str] = CONTEXT_FAMILIES,
) -> tuple[MultiDepthBracketExample, ...]:
    if int(contents) != int(fit_contents) + int(cal_contents) + int(test_contents):
        raise ValueError("contents must equal fit_contents + cal_contents + test_contents")
    depths = tuple(int(depth) for depth in depths)
    if not depths or min(depths) < 1:
        raise ValueError("depths must be positive")
    context_families = tuple(str(x) for x in context_families)
    if set(context_families) != set(CONTEXT_FAMILIES):
        raise ValueError("discovery bank must include all default context families")
    numeric_contents = deterministic_numeric_contents(int(contents))
    max_depth = max(depths)
    examples: list[MultiDepthBracketExample] = []
    for content_index, content in enumerate(numeric_contents):
        split = split_name_for_content_index(
            content_index,
            fit_contents=int(fit_contents),
            cal_contents=int(cal_contents),
            test_contents=int(test_contents),
        )
        for depth in depths:
            for context_family in context_families:
                prefix = context_prefix(context_family, depth=depth, max_depth=max_depth, content_index=content_index)
                tail = build_active_tail(depth, content)
                prompt = prefix + tail
                examples.append(
                    MultiDepthBracketExample(
                        example_id=f"{split}-c{content_index:03d}-d{depth}-{context_family}",
                        prompt=prompt,
                        token_ids=encode_prompt(enc, prompt),
                        tail=tail,
                        depth=int(depth),
                        close_count=close_count_from_depth(depth),
                        split=split,
                        pair_id=f"content-{content_index:03d}-{context_family}",
                        context_family=context_family,
                        numeric_content=content,
                        surface_open_count=prompt.count("["),
                        surface_close_count=prompt.count("]"),
                    )
                )
    return tuple(examples)


def bank_manifest(examples: Sequence[MultiDepthBracketExample]) -> dict[str, Any]:
    by_split: dict[str, list[MultiDepthBracketExample]] = defaultdict(list)
    for ex in examples:
        by_split[ex.split].append(ex)
    out: dict[str, Any] = {"total_examples": len(examples), "splits": {}}
    for split, rows in sorted(by_split.items()):
        indices = sorted({int(re.search(r"content-(\d+)-", ex.pair_id).group(1)) for ex in rows})
        out["splits"][split] = {
            "examples": len(rows),
            "content_count": len(indices),
            "content_indices": indices,
            "depths": sorted({ex.depth for ex in rows}),
            "context_families": sorted({ex.context_family for ex in rows}),
        }
    split_sets = [set(payload["content_indices"]) for payload in out["splits"].values()]
    out["content_splits_disjoint"] = all(not (a & b) for i, a in enumerate(split_sets) for b in split_sets[i + 1 :])
    return out


def r_value(example: MultiDepthBracketExample) -> float:
    return 1.0 if int(example.depth) >= 2 else 0.0


def relation_for_pair(
    base: MultiDepthBracketExample,
    source: MultiDepthBracketExample,
    relation: str,
) -> MultiDepthResamplingSpec | None:
    if base.example_id == source.example_id:
        return None
    same_d = int(base.depth) == int(source.depth)
    same_r = int(base.close_count) == int(source.close_count)
    if relation == "different_R":
        if not same_r:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "same_R_different_D":
        if same_r and not same_d:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "same_D":
        if same_d and base.pair_id != source.pair_id:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "same_surface_different_active_context":
        if (
            not same_d
            and base.surface_open_count == source.surface_open_count
            and base.context_family != source.context_family
        ):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "active_context")
    elif relation == "wrong_numeric_content":
        if same_d and base.context_family == source.context_family and numbers_only(base.tail) != numbers_only(source.tail):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "numeric_content")
    elif relation == "wrong_tail_length":
        if same_d and base.context_family == source.context_family and len(base.tail) != len(source.tail):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "tail_length")
    else:
        raise ValueError(f"unknown relation: {relation}")
    return None


def balanced_prefix(
    specs: Sequence[MultiDepthResamplingSpec],
    examples_by_id: Mapping[str, MultiDepthBracketExample],
    limit: int,
) -> list[MultiDepthResamplingSpec]:
    if int(limit) <= 0:
        return []
    if len(specs) <= int(limit):
        return list(specs)
    by_base_depth: dict[int, list[MultiDepthResamplingSpec]] = defaultdict(list)
    for spec in specs:
        by_base_depth[int(examples_by_id[spec.base_id].depth)].append(spec)
    out: list[MultiDepthResamplingSpec] = []
    depths = sorted(by_base_depth)
    while len(out) < int(limit) and any(by_base_depth.values()):
        for depth in depths:
            rows = by_base_depth[depth]
            if rows and len(out) < int(limit):
                out.append(rows.pop(0))
    return out


def build_relation_specs_for_split(
    examples: Sequence[MultiDepthBracketExample],
    *,
    split: str,
    records_per_relation: int,
    relations: Sequence[str] = DISCOVERY_RELATIONS,
) -> tuple[MultiDepthResamplingSpec, ...]:
    split_examples = sorted((ex for ex in examples if ex.split == split), key=lambda ex: ex.example_id)
    by_id = {ex.example_id: ex for ex in split_examples}
    selected: list[MultiDepthResamplingSpec] = []
    for relation in relations:
        rows: list[MultiDepthResamplingSpec] = []
        for base in split_examples:
            for source in split_examples:
                spec = relation_for_pair(base, source, relation)
                if spec is not None:
                    rows.append(spec)
        selected.extend(balanced_prefix(rows, by_id, int(records_per_relation)))
    return tuple(selected)


def relation_counts(specs: Iterable[MultiDepthResamplingSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec.relation] += 1
    return dict(sorted(counts.items()))


def masked_components(
    spec: MultiDepthResamplingSpec,
    components: Sequence[float],
    *,
    relations: Sequence[str] = DISCOVERY_RELATIONS,
) -> tuple[float, ...]:
    out: list[float] = []
    for relation in relations:
        if spec.relation == relation:
            out.extend(float(value) for value in components)
        else:
            out.extend(0.0 for _ in components)
    return tuple(out)


def feature_groups(
    specs: Sequence[MultiDepthResamplingSpec],
    component_names: Sequence[str],
    *,
    relations: Sequence[str] = DISCOVERY_RELATIONS,
) -> tuple[dict[str, Any], ...]:
    groups: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        for relation in relations:
            for component_index, component in enumerate(component_names):
                groups.append(
                    {
                        "spec_index": spec_index,
                        "actual_relation": spec.relation,
                        "block_relation": relation,
                        "component": component,
                        "component_index": component_index,
                    }
                )
    return tuple(groups)


def abstract_signature(
    stage: StageSpec,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
) -> tuple[float, ...]:
    values: list[float] = []
    for spec in specs:
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        r_delta = r_value(source) - r_value(base)
        values.extend(masked_components(spec, (r_delta,) * len(stage.component_names)))
    return tuple(values)


def read_jsonl_signatures(path: Path) -> dict[str, tuple[float, ...]]:
    if not path.exists():
        return {}
    out: dict[str, tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["site_id"])] = tuple(float(x) for x in row["signature"])
    return out


def append_jsonl_signature(path: Path, site_id: str, signature: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"site_id": str(site_id), "signature": [float(x) for x in signature]}, sort_keys=True) + "\n")


def handle_scalar(features: Mapping[str, float], weights_by_site: Mapping[str, float]) -> float:
    return float(sum(float(weight) * float(features[site_id]) for site_id, weight in weights_by_site.items()))


def equal_weights(site_ids: Sequence[str]) -> dict[str, float]:
    if not site_ids:
        return {}
    weight = 1.0 / float(len(site_ids))
    return {str(site_id): weight for site_id in site_ids}


def run_exact_replay(
    model: Any,
    *,
    base: CleanRun,
    target_features: Mapping[str, float],
    replay_sites: Sequence[ChannelSite],
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[float, int, dict[str, float], tuple[float, ...]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = _make_weighted_patch_from_features(
        replay_sites,
        source_features=target_features,
        position=base.final_position,
        weights_by_site={site.site_id: 1.0 for site in replay_sites},
        strength=1.0,
    )
    with torch.no_grad():
        with hook_recorder(regex=_hook_regex(record_sites), interventions=interventions) as ctx:
            logits, _, _ = model(base.token_ids)
    cache = {k: v.detach().cpu() for k, v in ctx.items()}
    features = _feature_dict(cache, record_sites, base.final_position)
    vector = _feature_vector(features, record_sites)
    margin = _bracket_margin(
        logits.detach().cpu(),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    return margin, _sign_from_margin(margin), features, vector


def stage_site_signature(
    *,
    model: Any,
    stage: StageSpec,
    site: ChannelSite,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[float, ...]:
    values: list[float] = []
    rlate_sites = tuple(site_lookup[site_id] for site_id in stage.frozen_readout_site_ids)
    rmid_sites = tuple(site_lookup[site_id] for site_id in stage.frozen_mid_site_ids)
    for spec in specs:
        base = clean_runs[spec.base_id]
        source = clean_runs[spec.source_id]
        direct_margin, _direct_pred, direct_features, _direct_vector = _run_weighted_patch(
            model,
            base=base,
            source=source,
            patch_sites=(site,),
            weights_by_site={site.site_id: 1.0},
            strength=1.0,
            record_sites=record_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        components: list[float] = [float(direct_margin - base.margin)]
        if "Rmid_replay_output_delta" in stage.component_names:
            margin, _pred, _features, _vector = run_exact_replay(
                model,
                base=base,
                target_features=direct_features,
                replay_sites=rmid_sites,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            components.append(float(margin - base.margin))
        if "Rlate_replay_output_delta" in stage.component_names:
            margin, _pred, _features, _vector = run_exact_replay(
                model,
                base=base,
                target_features=direct_features,
                replay_sites=rlate_sites,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            components.append(float(margin - base.margin))
        values.extend(masked_components(spec, components))
    return tuple(values)


def build_or_resume_stage_signatures(
    *,
    model: Any,
    stage: StageSpec,
    candidate_sites: Sequence[ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    checkpoint_path: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, tuple[float, ...]]:
    signatures = read_jsonl_signatures(checkpoint_path) if resume else {}
    if signatures:
        print(f"resuming {stage.stage_id} signatures: loaded {len(signatures)}/{len(candidate_sites)}", flush=True)
    for idx, site in enumerate(candidate_sites, start=1):
        if site.site_id in signatures:
            continue
        sig = stage_site_signature(
            model=model,
            stage=stage,
            site=site,
            specs=specs,
            examples=examples,
            clean_runs=clean_runs,
            site_lookup=site_lookup,
            record_sites=record_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        signatures[site.site_id] = sig
        append_jsonl_signature(checkpoint_path, site.site_id, sig)
        if idx % 10 == 0 or idx == len(candidate_sites):
            print(f"{stage.stage_id} signatures {idx}/{len(candidate_sites)}", flush=True)
    expected = {site.site_id for site in candidate_sites}
    actual = set(signatures)
    if actual != expected:
        raise ValueError(f"incomplete {stage.stage_id} signatures; missing={sorted(expected - actual)[:5]}")
    return signatures


def normalize_by_relation_component(
    abstract: Sequence[float],
    neural_by_site: Mapping[str, Sequence[float]],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], dict[str, float]]:
    abstract_tensor = torch.tensor(list(abstract), dtype=torch.float32)
    site_ids = tuple(neural_by_site)
    neural_tensor = torch.tensor([list(neural_by_site[site_id]) for site_id in site_ids], dtype=torch.float32)
    scales_by_key: dict[str, float] = {}
    for relation in DISCOVERY_RELATIONS:
        for component in sorted({str(group["component"]) for group in groups}):
            indices = [idx for idx, group in enumerate(groups) if group["block_relation"] == relation and group["component"] == component]
            if not indices:
                continue
            neural_rms = float(torch.sqrt((neural_tensor[:, indices] ** 2).mean()).item())
            abstract_rms = float(torch.sqrt((abstract_tensor[indices] ** 2).mean()).item())
            scale = max(neural_rms, abstract_rms, 1e-6)
            scales_by_key[f"{relation}:{component}"] = scale
            neural_tensor[:, indices] = neural_tensor[:, indices] / scale
            abstract_tensor[indices] = abstract_tensor[indices] / scale
    normalized_neural = {site_id: tuple(float(x) for x in neural_tensor[row_idx].tolist()) for row_idx, site_id in enumerate(site_ids)}
    return tuple(float(x) for x in abstract_tensor.tolist()), normalized_neural, scales_by_key


def selector_payload(
    *,
    stage: StageSpec,
    abstract: Sequence[float],
    neural_by_site: Mapping[str, Sequence[float]],
    candidate_sites: Sequence[ChannelSite],
    feature_groups_payload: Sequence[Mapping[str, Any]],
    epsilon: float,
    beta: float,
    block_normalize: bool,
) -> dict[str, Any]:
    if block_normalize:
        abstract, neural_by_site, scales = normalize_by_relation_component(abstract, neural_by_site, feature_groups_payload)
    else:
        scales = {}
    site_ids = tuple(site.site_id for site in candidate_sites)
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural_by_site[site_id]) for site_id in site_ids], dtype=torch.float32)
    cosine_cost = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    cosine_uot = sinkhorn_one_sided_uot(cosine_cost, epsilon=float(epsilon), beta_neural=float(beta), n_iter=300)[0]
    similarity = 1.0 - cosine_cost[0]
    direct = similarity.clamp_min(0.0)
    if float(direct.sum()) <= 0.0:
        direct = torch.softmax(similarity, dim=0)
    else:
        direct = direct / direct.sum().clamp_min(1e-12)
    squared_cost = cost_matrix(abstract_tensor, neural_tensor, mode="squared")
    squared_uot = sinkhorn_one_sided_uot(squared_cost, epsilon=float(epsilon), beta_neural=float(beta), n_iter=300)[0]

    def ranked_rows(weights: torch.Tensor, costs: torch.Tensor) -> list[dict[str, Any]]:
        rows = []
        for idx, site_id in enumerate(site_ids):
            rows.append({"site_id": site_id, "weight": float(weights[idx]), "cost": float(costs[idx])})
        return sorted(rows, key=lambda row: (-float(row["weight"]), float(row["cost"]), row["site_id"]))

    return {
        "stage_id": stage.stage_id,
        "component_names": list(stage.component_names),
        "candidate_count": len(site_ids),
        "abstract_signature_length": len(abstract),
        "block_normalize_signature": bool(block_normalize),
        "normalization_scales": scales,
        "feature_groups": list(feature_groups_payload),
        "selectors": {
            "raw_cosine_uot": {
                "cost_mode": "cosine",
                "coupling_rule": "one_sided_uot",
                "ranked_sites": ranked_rows(cosine_uot, cosine_cost[0]),
            },
            "raw_cosine_similarity": {
                "cost_mode": "cosine",
                "coupling_rule": "row_normalized_positive_cosine_similarity",
                "ranked_sites": ranked_rows(direct, cosine_cost[0]),
            },
            "raw_squared_uot": {
                "cost_mode": "squared",
                "coupling_rule": "one_sided_uot",
                "ranked_sites": ranked_rows(squared_uot, squared_cost[0]),
            },
        },
        "main_selector": "raw_cosine_uot",
    }


def weights_from_ranked(ranked_sites: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, float]:
    chosen = list(ranked_sites[: max(1, int(k))])
    total = sum(float(row.get("weight", 0.0)) for row in chosen)
    if total <= 0.0:
        return {str(row["site_id"]): 1.0 / len(chosen) for row in chosen}
    return {str(row["site_id"]): float(row["weight"]) / total for row in chosen}


def moves_toward_source(*, base: float, source: float, patched: float) -> bool:
    if abs(float(source) - float(base)) <= 1e-6:
        return False
    return abs(float(source) - float(patched)) + 1e-6 < abs(float(source) - float(base))


def effect_fraction(*, base: float, source: float, patched: float) -> float:
    denom = abs(float(source) - float(base))
    if denom <= 1e-6:
        return float("nan")
    return abs(float(patched) - float(base)) / denom


def safe_mean(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        val = float(value)
        if not math.isnan(val):
            vals.append(val)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def mean(values: Iterable[Any]) -> float:
    vals = [float(x) for x in values]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def validate_stage_handle(
    *,
    model: Any,
    stage: StageSpec,
    handle_id: str,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    rlate_sites = tuple(site_lookup[site_id] for site_id in stage.frozen_readout_site_ids)
    rmid_sites = tuple(site_lookup[site_id] for site_id in stage.frozen_mid_site_ids)
    rlate_weights = dict(stage.frozen_readout_weights or equal_weights(stage.frozen_readout_site_ids))
    rmid_weights = dict(stage.frozen_mid_weights or equal_weights(stage.frozen_mid_site_ids))
    records: list[dict[str, Any]] = []
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = clean_runs[spec.base_id]
        source = clean_runs[spec.source_id]
        patched_margin, patched_close_count, patched_features, _vector = _run_weighted_patch(
            model,
            base=base,
            source=source,
            patch_sites=patch_sites,
            weights_by_site=weights_by_site,
            strength=float(strength),
            record_sites=record_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        row: dict[str, Any] = {
            "handle_id": handle_id,
            "stage_id": stage.stage_id,
            "site_ids": [site.site_id for site in patch_sites],
            "weights_by_site": dict(weights_by_site),
            "strength": float(strength),
            "relation": spec.relation,
            "base_example_id": base_ex.example_id,
            "source_example_id": source_ex.example_id,
            "base_depth": base_ex.depth,
            "source_depth": source_ex.depth,
            "base_close_count": base_ex.close_count,
            "source_close_count": source_ex.close_count,
            "base_margin": base.margin,
            "source_margin": source.margin,
            "patched_margin": patched_margin,
            "patched_close_count": patched_close_count,
            "output_preserves_base": patched_close_count == base_ex.close_count,
            "output_matches_source": patched_close_count == source_ex.close_count,
            "output_flips": patched_close_count != base_ex.close_count,
        }
        if rlate_sites:
            base_rlate = handle_scalar(base.features_by_site, rlate_weights)
            source_rlate = handle_scalar(source.features_by_site, rlate_weights)
            patched_rlate = handle_scalar(patched_features, rlate_weights)
            replay_margin, replay_close_count, _features, _vec = run_exact_replay(
                model,
                base=base,
                target_features=patched_features,
                replay_sites=rlate_sites,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            row.update(
                {
                    "base_Rlate": base_rlate,
                    "source_Rlate": source_rlate,
                    "patched_Rlate": patched_rlate,
                    "Rlate_moves_toward_source": moves_toward_source(base=base_rlate, source=source_rlate, patched=patched_rlate),
                    "Rlate_effect_fraction": effect_fraction(base=base_rlate, source=source_rlate, patched=patched_rlate),
                    "Rlate_replay_margin": replay_margin,
                    "Rlate_replay_close_count": replay_close_count,
                    "Rlate_replay_matches_source": replay_close_count == source_ex.close_count,
                    "Rlate_replay_flips": replay_close_count != base_ex.close_count,
                }
            )
        if rmid_sites:
            base_rmid = handle_scalar(base.features_by_site, rmid_weights)
            source_rmid = handle_scalar(source.features_by_site, rmid_weights)
            patched_rmid = handle_scalar(patched_features, rmid_weights)
            replay_margin, replay_close_count, _features, _vec = run_exact_replay(
                model,
                base=base,
                target_features=patched_features,
                replay_sites=rmid_sites,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            row.update(
                {
                    "base_Rmid": base_rmid,
                    "source_Rmid": source_rmid,
                    "patched_Rmid": patched_rmid,
                    "Rmid_moves_toward_source": moves_toward_source(base=base_rmid, source=source_rmid, patched=patched_rmid),
                    "Rmid_effect_fraction": effect_fraction(base=base_rmid, source=source_rmid, patched=patched_rmid),
                    "Rmid_replay_margin": replay_margin,
                    "Rmid_replay_close_count": replay_close_count,
                    "Rmid_replay_matches_source": replay_close_count == source_ex.close_count,
                    "Rmid_replay_flips": replay_close_count != base_ex.close_count,
                }
            )
        records.append(row)
    summary = summarize_stage_records(stage.stage_id, records)
    acceptance = acceptance_for_stage(stage, summary, selected_site_ids=[site.site_id for site in patch_sites])
    return {"records": records, "summary": summary, "acceptance": acceptance, "score": score_for_stage(stage.stage_id, summary)}


def summarize_stage_records(stage_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)

    def rows(name: str) -> list[Mapping[str, Any]]:
        return by_relation.get(name, [])

    diff_r = rows("different_R")
    same_r_diff_d = rows("same_R_different_D")
    summary: dict[str, Any] = {
        "records": len(records),
        "relation_counts": {key: len(value) for key, value in sorted(by_relation.items())},
        "different_R_output_matches_source": mean(row["output_matches_source"] for row in diff_r),
        "different_R_output_flip": mean(row["output_flips"] for row in diff_r),
        "same_R_different_D_output_flip": mean(row["output_flips"] for row in same_r_diff_d),
        "same_D_output_preserve": mean(row["output_preserves_base"] for row in rows("same_D")),
        "same_surface_output_preserve": mean(row["output_preserves_base"] for row in rows("same_surface_different_active_context")),
        "wrong_numeric_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_numeric_content")),
        "wrong_tail_length_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_tail_length")),
    }
    for prefix in ("Rlate", "Rmid"):
        if any(f"{prefix}_replay_matches_source" in row for row in records):
            summary[f"different_R_{prefix}_moves"] = mean(row.get(f"{prefix}_moves_toward_source", False) for row in diff_r)
            summary[f"different_R_{prefix}_fraction"] = safe_mean(row.get(f"{prefix}_effect_fraction", float("nan")) for row in diff_r)
            summary[f"different_R_{prefix}_replay_matches_source"] = mean(row.get(f"{prefix}_replay_matches_source", False) for row in diff_r)
            summary[f"same_R_different_D_{prefix}_moves"] = mean(row.get(f"{prefix}_moves_toward_source", False) for row in same_r_diff_d)
            summary[f"same_R_different_D_{prefix}_replay_flips"] = mean(row.get(f"{prefix}_replay_flips", False) for row in same_r_diff_d)
    return summary


def support_collapses(selected: Sequence[str], forbidden: Sequence[str]) -> bool:
    selected_set = set(selected)
    forbidden_set = set(forbidden)
    return bool(selected_set) and selected_set <= forbidden_set


def acceptance_for_stage(stage: StageSpec, summary: Mapping[str, Any], *, selected_site_ids: Sequence[str]) -> dict[str, Any]:
    threshold = 0.90
    if stage.stage_id.startswith("stageA"):
        checks = {
            "different_R_output_matches_source": float(summary.get("different_R_output_matches_source", float("nan"))) >= threshold,
            "different_R_Rlate_moves": float(summary.get("different_R_Rlate_moves", float("nan"))) >= threshold,
            "same_R_different_D_output_flip_le_0.10": float(summary.get("same_R_different_D_output_flip", float("nan"))) <= 0.10,
            "wrong_numeric_output_preserve": float(summary.get("wrong_numeric_output_preserve", float("nan"))) >= threshold,
            "wrong_tail_length_output_preserve": float(summary.get("wrong_tail_length_output_preserve", float("nan"))) >= threshold,
        }
        return {"validated": all(checks.values()), "checks": checks}
    if stage.stage_id.startswith("stageB"):
        forbidden = set(stage.frozen_readout_site_ids)
        checks = {
            "different_R_Rlate_replay_matches_source": float(summary.get("different_R_Rlate_replay_matches_source", float("nan"))) >= threshold,
            "different_R_output_matches_source": float(summary.get("different_R_output_matches_source", float("nan"))) >= threshold,
            "same_R_different_D_Rlate_moves_le_0.10": float(summary.get("same_R_different_D_Rlate_moves", float("nan"))) <= 0.10,
            "same_R_different_D_Rlate_replay_flips_le_0.10": float(summary.get("same_R_different_D_Rlate_replay_flips", float("nan"))) <= 0.10,
            "same_R_different_D_output_flip_le_0.10": float(summary.get("same_R_different_D_output_flip", float("nan"))) <= 0.10,
            "wrong_numeric_output_preserve": float(summary.get("wrong_numeric_output_preserve", float("nan"))) >= threshold,
            "wrong_tail_length_output_preserve": float(summary.get("wrong_tail_length_output_preserve", float("nan"))) >= threshold,
            "support_not_Rlate": not support_collapses(selected_site_ids, tuple(forbidden)),
        }
        return {"validated": all(checks.values()), "R_mid_validated": all(checks.values()), "checks": checks}
    if stage.stage_id.startswith("stageC"):
        forbidden = set(stage.frozen_readout_site_ids) | set(stage.frozen_mid_site_ids)
        checks = {
            "stageB_validated_required": bool(stage.acceptance_eligible),
            "different_R_Rmid_replay_matches_source": float(summary.get("different_R_Rmid_replay_matches_source", float("nan"))) >= threshold,
            "different_R_Rlate_replay_matches_source": float(summary.get("different_R_Rlate_replay_matches_source", float("nan"))) >= threshold,
            "different_R_output_matches_source": float(summary.get("different_R_output_matches_source", float("nan"))) >= threshold,
            "same_R_different_D_Rmid_moves_le_0.10": float(summary.get("same_R_different_D_Rmid_moves", float("nan"))) <= 0.10,
            "same_R_different_D_Rmid_replay_flips_le_0.10": float(summary.get("same_R_different_D_Rmid_replay_flips", float("nan"))) <= 0.10,
            "same_R_different_D_Rlate_moves_le_0.10": float(summary.get("same_R_different_D_Rlate_moves", float("nan"))) <= 0.10,
            "same_R_different_D_Rlate_replay_flips_le_0.10": float(summary.get("same_R_different_D_Rlate_replay_flips", float("nan"))) <= 0.10,
            "same_R_different_D_output_flip_le_0.10": float(summary.get("same_R_different_D_output_flip", float("nan"))) <= 0.10,
            "wrong_numeric_output_preserve": float(summary.get("wrong_numeric_output_preserve", float("nan"))) >= threshold,
            "wrong_tail_length_output_preserve": float(summary.get("wrong_tail_length_output_preserve", float("nan"))) >= threshold,
            "support_not_Rmid_or_Rlate": not support_collapses(selected_site_ids, tuple(forbidden)),
        }
        return {"validated": all(checks.values()), "R_pre_validated": all(checks.values()), "checks": checks}
    raise ValueError(stage.stage_id)


def score_for_stage(stage_id: str, summary: Mapping[str, Any]) -> float:
    if stage_id.startswith("stageA"):
        values = [
            float(summary.get("different_R_output_matches_source", 0.0)),
            float(summary.get("different_R_Rlate_moves", 0.0)),
            1.0 - float(summary.get("same_R_different_D_output_flip", 1.0)),
            float(summary.get("wrong_numeric_output_preserve", 0.0)),
            float(summary.get("wrong_tail_length_output_preserve", 0.0)),
        ]
    elif stage_id.startswith("stageB"):
        values = [
            float(summary.get("different_R_Rlate_replay_matches_source", 0.0)),
            float(summary.get("different_R_output_matches_source", 0.0)),
            1.0 - float(summary.get("same_R_different_D_Rlate_moves", 1.0)),
            1.0 - float(summary.get("same_R_different_D_Rlate_replay_flips", 1.0)),
            1.0 - float(summary.get("same_R_different_D_output_flip", 1.0)),
            float(summary.get("wrong_numeric_output_preserve", 0.0)),
            float(summary.get("wrong_tail_length_output_preserve", 0.0)),
        ]
    elif stage_id.startswith("stageC"):
        values = [
            float(summary.get("different_R_Rmid_replay_matches_source", 0.0)),
            float(summary.get("different_R_Rlate_replay_matches_source", 0.0)),
            float(summary.get("different_R_output_matches_source", 0.0)),
            1.0 - float(summary.get("same_R_different_D_Rmid_moves", 1.0)),
            1.0 - float(summary.get("same_R_different_D_Rmid_replay_flips", 1.0)),
            1.0 - float(summary.get("same_R_different_D_Rlate_moves", 1.0)),
            1.0 - float(summary.get("same_R_different_D_Rlate_replay_flips", 1.0)),
            1.0 - float(summary.get("same_R_different_D_output_flip", 1.0)),
            float(summary.get("wrong_numeric_output_preserve", 0.0)),
            float(summary.get("wrong_tail_length_output_preserve", 0.0)),
        ]
    else:
        raise ValueError(stage_id)
    return float(sum(0.0 if math.isnan(value) else value for value in values) / len(values))


def support_overlap(site_ids: Sequence[str]) -> dict[str, Any]:
    site_set = set(str(site_id) for site_id in site_ids)
    return {
        "includes_R1079": "4.attn.resid_delta:1079" in site_set,
        "includes_D1249": "2.attn.resid_delta:1249" in site_set,
        "includes_Rlate": sorted(site_set & set(H_RLATE_SITE_IDS)),
    }


def calibrate_stage(
    *,
    model: Any,
    stage: StageSpec,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_lookup: Mapping[str, ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    record_sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    lambda_grid: Sequence[float],
    single_close_token_id: int,
    double_close_token_id: int,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_metadata = {
        "stage_id": stage.stage_id,
        "k_grid": [int(k) for k in k_grid],
        "lambda_grid": [float(strength) for strength in lambda_grid],
        "ranked_sites": [
            {"site_id": str(row["site_id"]), "weight": float(row.get("weight", 0.0))}
            for row in ranked_sites
        ],
        "specs": [
            {
                "relation": str(spec.relation),
                "base_id": str(spec.base_id),
                "source_id": str(spec.source_id),
                "wrong_variable": None if spec.wrong_variable is None else str(spec.wrong_variable),
            }
            for spec in specs
        ],
    }
    rows: list[dict[str, Any]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("metadata") != checkpoint_metadata:
            raise ValueError(f"calibration checkpoint metadata mismatch: {checkpoint_path}")
        rows = list(checkpoint.get("rows", []))
        print(f"resuming {stage.stage_id} calibration: loaded {len(rows)} grid rows", flush=True)

    completed_handle_ids = {str(row["handle_id"]) for row in rows}
    for k in k_grid:
        weights = weights_from_ranked(ranked_sites, k=int(k))
        patch_sites = tuple(site_lookup[site_id] for site_id in weights)
        for strength in lambda_grid:
            handle_id = f"{stage.stage_id}_top{k}_lambda{float(strength):g}"
            if handle_id in completed_handle_ids:
                continue
            validation = validate_stage_handle(
                model=model,
                stage=stage,
                handle_id=handle_id,
                patch_sites=patch_sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
                examples=examples,
                clean_runs=clean_runs,
                site_lookup=site_lookup,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            rows.append(
                {
                    "handle_id": handle_id,
                    "k": int(k),
                    "strength": float(strength),
                    "site_ids": list(weights),
                    "weights_by_site": weights,
                    "summary": validation["summary"],
                    "acceptance": validation["acceptance"],
                    "calibration_score": float(validation["score"]),
                    "overlap": support_overlap(list(weights)),
                }
            )
            completed_handle_ids.add(handle_id)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                checkpoint_tmp.write_text(
                    json.dumps({"metadata": checkpoint_metadata, "rows": rows}, indent=2) + "\n",
                    encoding="utf-8",
                )
                checkpoint_tmp.replace(checkpoint_path)
        print(f"calibrated {stage.stage_id} K={k}", flush=True)
    best = sorted(rows, key=lambda row: (-float(row["calibration_score"]), int(row["k"]), abs(float(row["strength"]) - 1.0)))[0]
    return {"grid": rows, "best": best}


def tie_support(calibration: Mapping[str, Any], *, epsilon_optimal: float, exact_tie_tolerance: float = 1e-12) -> dict[str, Any]:
    rows = list(calibration["grid"])
    best_score = float(calibration["best"]["calibration_score"])
    exact = [row for row in rows if best_score - float(row["calibration_score"]) <= float(exact_tie_tolerance)]
    near = [row for row in rows if best_score - float(row["calibration_score"]) <= float(epsilon_optimal)]

    def summarize(rows_for_support: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        union = sorted({site_id for row in rows_for_support for site_id in row["site_ids"]})
        return {
            "handle_count": len(rows_for_support),
            "support_union": union,
            "includes_R1079": "4.attn.resid_delta:1079" in union,
            "includes_D1249": "2.attn.resid_delta:1249" in union,
            "includes_Rlate": sorted(set(union) & set(H_RLATE_SITE_IDS)),
            "handles": [
                {
                    "handle_id": row["handle_id"],
                    "k": row["k"],
                    "strength": row["strength"],
                    "calibration_score": row["calibration_score"],
                    "site_ids": row["site_ids"],
                    "overlap": row["overlap"],
                }
                for row in rows_for_support
            ],
        }

    return {
        "best_score": best_score,
        "exact_tie_support": summarize(exact),
        "epsilon_optimal_support": summarize(near),
        "epsilon_optimal": float(epsilon_optimal),
        "selection_rule": "Dcal only; Dte is not used for tie or epsilon-optimal support.",
    }


def posthoc_ranks(ranked_sites: Sequence[Mapping[str, Any]], site_ids: Sequence[str] = POSTHOC_SITE_IDS) -> dict[str, Any]:
    ranks = {}
    for target in site_ids:
        rank = None
        for idx, row in enumerate(ranked_sites, start=1):
            if str(row["site_id"]) == str(target):
                rank = idx
                break
        ranks[target] = rank
    return ranks


def bootstrap_support(
    *,
    stage: StageSpec,
    reps: int,
    seed: int,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    neural_by_site: Mapping[str, Sequence[float]],
    candidate_sites: Sequence[ChannelSite],
    epsilon: float,
    beta: float,
    block_normalize: bool,
) -> list[dict[str, Any]]:
    if int(reps) <= 0:
        return []
    by_relation: dict[str, list[MultiDepthResamplingSpec]] = defaultdict(list)
    for spec in specs:
        by_relation[spec.relation].append(spec)
    rng = random.Random(int(seed))
    records: list[dict[str, Any]] = []
    original_indices = {id(spec): idx for idx, spec in enumerate(specs)}
    component_count = len(stage.component_names)
    block_width = len(DISCOVERY_RELATIONS) * component_count
    for rep in range(int(reps)):
        sampled_specs: list[MultiDepthResamplingSpec] = []
        sampled_indices: list[int] = []
        for relation in sorted(by_relation):
            rows = by_relation[relation]
            for _ in range(len(rows)):
                spec = rng.choice(rows)
                sampled_specs.append(spec)
                sampled_indices.append(original_indices[id(spec)])
        feature_idx: list[int] = []
        for spec_idx in sampled_indices:
            start = spec_idx * block_width
            feature_idx.extend(range(start, start + block_width))
        sampled_neural = {
            site_id: tuple(float(sig[idx]) for idx in feature_idx)
            for site_id, sig in neural_by_site.items()
        }
        sampled_abstract_full = abstract_signature(stage, specs, examples)
        sampled_abstract = tuple(float(sampled_abstract_full[idx]) for idx in feature_idx)
        sampled_groups_full = feature_groups(specs, stage.component_names)
        sampled_groups = tuple(sampled_groups_full[idx] for idx in feature_idx)
        selector = selector_payload(
            stage=stage,
            abstract=sampled_abstract,
            neural_by_site=sampled_neural,
            candidate_sites=candidate_sites,
            feature_groups_payload=sampled_groups,
            epsilon=float(epsilon),
            beta=float(beta),
            block_normalize=block_normalize,
        )
        ranked = selector["selectors"]["raw_cosine_uot"]["ranked_sites"]
        top1 = [row["site_id"] for row in ranked[:1]]
        top2 = [row["site_id"] for row in ranked[:2]]
        records.append(
            {
                "rep": rep,
                "top1": top1,
                "top2": top2,
                "R1079_top1": "4.attn.resid_delta:1079" in top1,
                "R1079_top2": "4.attn.resid_delta:1079" in top2,
                "D1249_top1": "2.attn.resid_delta:1249" in top1,
                "D1249_top2": "2.attn.resid_delta:1249" in top2,
            }
        )
    return records


def summarize_bootstrap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": 0}

    def rate(key: str) -> float:
        return mean(row.get(key, False) for row in records)

    return {
        "records": len(records),
        "R1079_top1_rate": rate("R1079_top1"),
        "R1079_top2_rate": rate("R1079_top2"),
        "D1249_top1_rate": rate("D1249_top1"),
        "D1249_top2_rate": rate("D1249_top2"),
    }


def layer_order(site_id: str) -> tuple[int, int]:
    hook = site_id.split(":", 1)[0]
    if hook == "final_resid":
        return (999, 0)
    parts = hook.split(".")
    try:
        layer = int(parts[0])
    except ValueError:
        return (998, 0)
    hook_tail = ".".join(parts[1:])
    stage_order = {
        "resid_in": 0,
        "attn.act_in": 1,
        "attn.q": 2,
        "attn.k": 2,
        "attn.v": 2,
        "attn.y": 3,
        "attn.resid_delta": 4,
        "resid_mid": 5,
        "mlp.act_in": 6,
        "mlp.post_act": 7,
        "mlp.resid_delta": 8,
    }.get(hook_tail, 9)
    return (layer, stage_order)


def upstream_sites(candidate_sites: Sequence[ChannelSite], downstream_site_ids: Sequence[str]) -> tuple[ChannelSite, ...]:
    if not downstream_site_ids:
        return tuple(candidate_sites)
    cutoff = min(layer_order(site_id) for site_id in downstream_site_ids)
    return tuple(site for site in candidate_sites if layer_order(site.site_id) < cutoff)


def final_model(stage_a: Mapping[str, Any], stage_b: Mapping[str, Any], stage_c: Mapping[str, Any] | None) -> dict[str, Any]:
    b0 = bool(stage_a.get("heldout", {}).get("acceptance", {}).get("validated", False))
    b1 = bool(stage_b.get("heldout", {}).get("acceptance", {}).get("validated", False) and stage_b.get("mediation", {}).get("acceptance", {}).get("validated", False))
    b2 = bool(
        b1
        and stage_c
        and stage_c.get("heldout", {}).get("acceptance", {}).get("validated", False)
        and stage_c.get("mediation", {}).get("acceptance", {}).get("validated", False)
    )
    accepted = "none"
    if b2:
        accepted = "B2"
    elif b1:
        accepted = "B1"
    elif b0:
        accepted = "B0"
    return {
        "B0_X_to_Rlate_to_Y": {"accepted": b0},
        "B1_X_to_Rmid_to_Rlate_to_Y": {"accepted": b1},
        "B2_X_to_Rpre_to_Rmid_to_Rlate_to_Y": {"accepted": b2},
        "final_accepted_model": accepted,
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Bracket Progressive Model Discovery",
        "",
        "Search policy:",
        f"- candidate count: `{payload['candidate_manifest']['candidate_count']}`",
        f"- no filtering applied in all-133 audits: `{payload['candidate_manifest']['no_filtering_applied']}`",
        f"- bank examples: `{payload['bank_manifest']['total_examples']}`",
        f"- records per relation: `{payload['records_per_relation']}`",
        f"- clean accuracy: `{payload['clean']['accuracy']:.3f}`",
        "",
        "Accepted model:",
        f"- `{payload['model_lattice']['final_accepted_model']}`",
        "",
        "## Stages",
        "",
    ]
    for key in (
        "stageA_Rlate_from_output",
        "stageB_Rmid_from_Rlate_all133_audit",
        "stageB_Rmid_from_Rlate_primary",
        "stageB_Rmid_from_Rlate_upstream_only_secondary",
        "stageC_Rpre_from_Rmid_Rlate_all133_audit",
        "stageC_Rpre_from_Rmid_Rlate_primary",
        "stageC_Rpre_from_Rmid_Rlate_upstream_only_secondary",
    ):
        stage = payload.get(key)
        if not stage:
            continue
        best = stage["calibration"]["best"]
        held = stage["heldout"]
        lines.extend(
            [
                f"### `{key}`",
                "",
                f"- candidates: `{stage['candidate_count']}`",
                f"- best sites: `{', '.join(best['site_ids'])}`",
                f"- K: `{best['k']}`",
                f"- lambda: `{float(best['strength']):.3f}`",
                f"- Dcal score: `{float(best['calibration_score']):.3f}`",
                f"- Dte score: `{float(held['score']):.3f}`",
                f"- heldout validated: `{held['acceptance']['validated']}`",
                f"- posthoc ranks: `{stage['posthoc_ranks']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
