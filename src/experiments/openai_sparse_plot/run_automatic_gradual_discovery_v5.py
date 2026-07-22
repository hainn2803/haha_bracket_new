from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ablate_rediscover import (
    ClampedRun,
    HandleConfiguration,
    RediscoveryExample,
    RediscoveryPair,
    abstract_signature,
    atomic_json,
    bank_manifest,
    build_bracket_pairs,
    build_bracket_rediscovery_bank,
    clean_accuracy,
    collect_clamped_runs,
    evaluate_configurations,
    load_candidate_circuit,
    match_signatures,
    relation_summary,
    select_calibration_row,
)
from .activation import ChannelSite
from .bracket_progressive_model_discovery import layer_order, upstream_sites
from .graded_evidence import abstract_e_signature, build_graded_evidence_bank, build_graded_pairs, decoder_metrics, e_value, fit_affine_decoder, graded_validation_summary
from .progressive_rearly import evaluate_progressive_configurations, fit_binary_scalar_readout, mediation_summary
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


SPLITS = ("Dfit", "Dcal", "Dte")
SELECTION_SPLITS = ("Dfit", "Dcal")


@dataclass
class Data:
    examples: tuple[RediscoveryExample, ...]
    pairs: dict[str, tuple[RediscoveryPair, ...]]
    by_id: dict[str, RediscoveryExample]
    runs: Mapping[str, ClampedRun]


@dataclass
class Context:
    args: argparse.Namespace
    model: Any
    sites: tuple[ChannelSite, ...]
    site_lookup: dict[str, ChannelSite]
    negative_token_id: int
    positive_token_id: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic gradual PLOT for X -> D -> R -> Y.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/automatic_gradual_discovery"))
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def parse_numbers(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value.strip())


def run_output(ctx: Context, data: Data, configs: Sequence[HandleConfiguration], split: str) -> np.ndarray:
    return evaluate_configurations(ctx.model, configs, data.pairs[split], examples=data.by_id, runs=data.runs, site_lookup=ctx.site_lookup, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)


def run_probes(ctx: Context, data: Data, configs: Sequence[HandleConfiguration], split: str, probe_sites: Sequence[ChannelSite], restore: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
    return evaluate_progressive_configurations(ctx.model, configs, data.pairs[split], examples=data.by_id, runs=data.runs, site_lookup=ctx.site_lookup, probe_sites=probe_sites, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size, restore_probe_site_ids=restore)


def make_data(ctx: Context, examples: tuple[RediscoveryExample, ...], pairs: dict[str, tuple[RediscoveryPair, ...]]) -> Data:
    runs = collect_clamped_runs(ctx.model, examples, candidate_sites=ctx.sites, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)
    return Data(examples, pairs, {row.example_id: row for row in examples}, runs)


def normalize_blocks(abstract: Sequence[float], neural: Mapping[str, Sequence[float]], pairs: Sequence[RediscoveryPair], components: int) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], dict[str, float]]:
    abstract_tensor = torch.tensor(abstract, dtype=torch.float32)
    site_ids = tuple(neural)
    neural_tensor = torch.tensor([neural[site_id] for site_id in site_ids], dtype=torch.float32)
    scales = {}
    for relation in sorted({pair.relation for pair in pairs}):
        for component in range(components):
            indices = [components * index + component for index, pair in enumerate(pairs) if pair.relation == relation]
            scale = max(float(torch.sqrt((abstract_tensor[indices] ** 2).mean())), float(torch.sqrt((neural_tensor[:, indices] ** 2).mean())), 1e-6)
            abstract_tensor[indices] /= scale
            neural_tensor[:, indices] /= scale
            scales[f"{relation}:{component}"] = scale
    normalized = {site_id: tuple(float(value) for value in neural_tensor[index]) for index, site_id in enumerate(site_ids)}
    return tuple(float(value) for value in abstract_tensor), normalized, scales


def get_support(ranked: Sequence[Mapping[str, Any]], top_n: int, mass_fraction: float) -> list[dict[str, Any]]:
    rows = [dict(row) for row in ranked[:top_n]]
    if not rows:
        return []
    cutoff = mass_fraction * float(rows[0]["weight"])
    return [row for row in rows if float(row["weight"]) >= cutoff]


def make_calibration_configs(support: Sequence[Mapping[str, Any]], strengths: Sequence[float]) -> tuple[tuple[HandleConfiguration, ...], list[dict[str, Any]]]:
    configs = []
    metadata = []
    for size in (1, 2):
        for index, combo in enumerate(combinations(support, size), start=1):
            total = sum(float(row["weight"]) for row in combo)
            weights = {str(row["site_id"]): float(row["weight"]) / total for row in combo}
            for strength in strengths:
                handle_id = f"S{size}_{index}_lambda{strength:g}"
                configs.append(HandleConfiguration(handle_id, weights, strength))
                metadata.append({"handle_id": handle_id, "k": size, "strength": strength, "weights": weights, "site_ids": list(weights)})
    return tuple(configs), metadata


def get_best_tier(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_score = max(float(row["summary"]["score"]) for row in rows)
    tier = [dict(row) for row in rows if abs(float(row["summary"]["score"]) - best_score) < 1e-12]
    return dict(select_calibration_row(tier)), tier


def best_strength_per_handle(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row["site_ids"])].append(row)
    selected = []
    for group in groups.values():
        best = sorted(group, key=lambda row: (-float(row["summary"]["score"]), abs(float(row["strength"]) - 1.0)))[0]
        if best["summary"]["passes"]:
            selected.append(dict(best))
    return selected


def handle_value(run: ClampedRun, weights: Mapping[str, float]) -> float:
    return sum(float(weight) * float(run.features_by_site[site_id]) for site_id, weight in weights.items())


def handle_order(weights: Mapping[str, float]) -> tuple[int, int]:
    return max(layer_order(site_id) for site_id in weights)


def get_probe_sites(ctx: Context, *handles: Mapping[str, float]) -> tuple[ChannelSite, ...]:
    site_ids = []
    for handle in handles:
        for site_id in handle:
            if site_id not in site_ids:
                site_ids.append(site_id)
    return tuple(ctx.site_lookup[site_id] for site_id in site_ids)


def combine_probes(probes: np.ndarray, probe_ids: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    index = {site_id: position for position, site_id in enumerate(probe_ids)}
    return sum(float(weight) * probes[..., index[site_id]] for site_id, weight in weights.items())


def fit_r(examples: Sequence[RediscoveryExample], runs: Mapping[str, ClampedRun], weights: Mapping[str, float], splits: Sequence[str] = SPLITS) -> tuple[Any, dict[str, float]]:
    fit_rows = [row for row in examples if row.split == "Dfit"]
    readout = fit_binary_scalar_readout([handle_value(runs[row.example_id], weights) for row in fit_rows], [row.variable_value for row in fit_rows])
    accuracy = {split: float(np.mean([readout.predict(handle_value(runs[row.example_id], weights)) == row.variable_value for row in examples if row.split == split])) for split in splits}
    return readout, accuracy


def fit_d(examples: Sequence[RediscoveryExample], runs: Mapping[str, ClampedRun], weights: Mapping[str, float], splits: Sequence[str] = SPLITS) -> tuple[Any, dict[str, dict[str, float]]]:
    fit_rows = [row for row in examples if row.split == "Dfit"]
    decoder = fit_affine_decoder([handle_value(runs[row.example_id], weights) for row in fit_rows], [e_value(row, "active_depth") for row in fit_rows])
    metrics = {}
    for split in splits:
        rows = [row for row in examples if row.split == split]
        values = [handle_value(runs[row.example_id], weights) for row in rows]
        targets = [e_value(row, "active_depth") for row in rows]
        metrics[split] = decoder_metrics(decoder, values, targets)
    return decoder, metrics


def is_graded_d(metrics: Mapping[str, Mapping[str, float]], threshold: float, splits: Sequence[str] = SELECTION_SPLITS) -> bool:
    return min(abs(float(metrics[split]["pearson"])) for split in splits) >= threshold


def classify_handles(rows: Sequence[Mapping[str, Any]], graded: Data, threshold: float, splits: Sequence[str] = SELECTION_SPLITS) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in rows:
        weights = {str(site_id): float(weight) for site_id, weight in row["weights"].items()}
        key = tuple(sorted(weights.items()))
        if key in seen:
            continue
        seen.add(key)
        r_readout, r_accuracy = fit_r(graded.examples, graded.runs, weights, splits)
        d_decoder, d_metrics = fit_d(graded.examples, graded.runs, weights, splits)
        result.append({"row": dict(row), "weights": weights, "r_readout": r_readout, "r_accuracy": r_accuracy, "d_decoder": d_decoder, "d_metrics": d_metrics, "is_D": is_graded_d(d_metrics, threshold, splits)})
    return result


def public_handle(handle: Mapping[str, Any]) -> dict[str, Any]:
    return {"handle_id": handle["row"]["handle_id"], "weights": handle["weights"], "strength": handle["row"]["strength"], "R_accuracy": handle["r_accuracy"], "D_decoder": handle["d_decoder"].to_dict(), "D_metrics": handle["d_metrics"], "is_D": handle["is_D"]}


def expected_r(pair: RediscoveryPair, examples: Mapping[str, RediscoveryExample]) -> int:
    base = examples[pair.base_id].variable_value
    source = examples[pair.source_id].variable_value
    return int(source if source != base else base)


def r_summary(data: Data, split: str, margins: Sequence[float], downstream: Sequence[float], frozen_weights: Mapping[str, float], readout: Any) -> dict[str, Any]:
    by_relation: dict[str, list[dict[str, bool]]] = defaultdict(list)
    sensitivity = []
    invariance = []
    for index, pair in enumerate(data.pairs[split]):
        base = handle_value(data.runs[pair.base_id], frozen_weights)
        source = handle_value(data.runs[pair.source_id], frozen_weights)
        patched = float(downstream[index])
        row = {"output_correct": (1 if float(margins[index]) > 0 else -1) == expected_r(pair, data.by_id), "downstream_correct": readout.predict(patched) == expected_r(pair, data.by_id), "downstream_moves": abs(source - patched) < abs(source - base) if abs(source - base) > 1e-8 else abs(patched - base) <= 1e-6}
        by_relation[pair.relation].append(row)
        target = sensitivity if data.by_id[pair.source_id].variable_value != data.by_id[pair.base_id].variable_value else invariance
        target.append(row)
    relations = {relation: {key: float(np.mean([row[key] for row in rows])) for key in ("output_correct", "downstream_correct", "downstream_moves")} for relation, rows in sorted(by_relation.items())}
    blocks = {"sensitivity_output": float(np.mean([row["output_correct"] for row in sensitivity])), "sensitivity_downstream": float(np.mean([row["downstream_correct"] for row in sensitivity])), "invariance_output": float(np.mean([row["output_correct"] for row in invariance])), "invariance_downstream": float(np.mean([row["downstream_correct"] for row in invariance]))}
    return {"relations": relations, "balanced_blocks": blocks, "score": float(np.mean(list(blocks.values()))), "passes": min(blocks.values()) >= 0.9}


def d_summary(data: Data, split: str, margins: Sequence[float], probes: np.ndarray, probe_ids: Sequence[str], d_weights: Mapping[str, float], d_decoder: Any, r_weights: Mapping[str, float], r_readout: Any) -> dict[str, Any]:
    abstract_d = {example_id: e_value(data.by_id[example_id], "active_depth") for example_id in data.runs}
    clean_d = {example_id: d_decoder.predict(handle_value(run, d_weights)) for example_id, run in data.runs.items()}
    patched_d = d_decoder.slope * combine_probes(probes, probe_ids, d_weights) + d_decoder.intercept
    patched_r = [r_readout.predict(value) for value in combine_probes(probes, probe_ids, r_weights)]
    patched_y = np.where(np.asarray(margins) > 0, 1, -1)
    return graded_validation_summary(data.pairs[split], data.by_id, abstract_d, clean_d, patched_d, patched_r, patched_y)


def neural_signature(data: Data, split: str, margins: Sequence[float], downstream: Sequence[float], frozen_weights: Mapping[str, float], scale: float, include_output: bool) -> tuple[float, ...]:
    signature = []
    for index, pair in enumerate(data.pairs[split]):
        signature.append(scale * (float(downstream[index]) - handle_value(data.runs[pair.base_id], frozen_weights)))
        if include_output:
            signature.append(float(margins[index]) - data.runs[pair.base_id].class_margin)
    return tuple(signature)


def repeated_r_signature(data: Data, split: str, components: int) -> tuple[float, ...]:
    signature = []
    for pair in data.pairs[split]:
        delta = float(data.by_id[pair.source_id].variable_value - data.by_id[pair.base_id].variable_value)
        signature.extend([delta] * components)
    return tuple(signature)


# Stage 1: X -> R -> Y.
def coarse_phase(ctx: Context, data: Data, strengths: Sequence[float]) -> dict[str, Any]:
    print("[1] Coarse X -> R -> Y", flush=True)

    # Dfit: compute one effect signature for every candidate site.
    singletons = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in ctx.sites)
    fit_margins = run_output(ctx, data, singletons, "Dfit")
    neural = {config.handle_id: tuple(float(fit_margins[index, pair_index] - data.runs[pair.base_id].class_margin) for pair_index, pair in enumerate(data.pairs["Dfit"])) for index, config in enumerate(singletons)}
    abstract, neural, scales = normalize_blocks(abstract_signature(data.pairs["Dfit"], data.by_id), neural, data.pairs["Dfit"], 1)
    selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
    support = get_support(selector["ranked"], ctx.args.top_n, ctx.args.mass_fraction)

    # Dcal: test every singleton and pair in the effective support.
    configs, metadata = make_calibration_configs(support, strengths)
    cal_margins = run_output(ctx, data, configs, "Dcal")
    grid = [{**row, "summary": relation_summary(data.pairs["Dcal"], data.by_id, cal_margins[index])} for index, row in enumerate(metadata)]
    best, tier = get_best_tier(grid)

    # Dte: test every configuration tied in the best calibration tier.
    test_configs = tuple(HandleConfiguration(row["handle_id"], row["weights"], row["strength"]) for row in tier)
    test_margins = run_output(ctx, data, test_configs, "Dte")
    heldout = [{**row, "heldout": relation_summary(data.pairs["Dte"], data.by_id, test_margins[index])} for index, row in enumerate(tier)]
    accepted = [row for row in heldout if row["heldout"]["all_rates_at_least_0_90"]]

    if not accepted:
        raise RuntimeError("no coarse handle passed Dte")
    return {"normalization_scales": scales, "selector": selector, "effective_support": support, "calibration_best": best, "calibration_best_tier": tier, "heldout_tier": heldout, "accepted": accepted}


def find_earlier_sites(sites: Sequence[ChannelSite], downstream_weights: Mapping[str, float]) -> tuple[ChannelSite, ...]:
    cutoff = handle_order(downstream_weights)
    return tuple(site for site in sites if site.site_id not in downstream_weights and layer_order(site.site_id) < cutoff)


def is_certified(summary: Mapping[str, Any], mediation: Mapping[str, Any]) -> bool:
    return bool(summary["passes"] and mediation["passes"])


# def rank_against_frozen_r(ctx: Context, data: Data, candidates: Sequence[ChannelSite], frozen_r: Mapping[str, Any]) -> dict[str, Any]:
#     probes = get_probe_sites(ctx, frozen_r["weights"])
#     probe_ids = tuple(site.site_id for site in probes)
#     configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidates)
#     margins, values = run_probes(ctx, data, configs, "Dfit", probes)
#     neural = {config.handle_id: neural_signature(data, "Dfit", margins[index], combine_probes(values[index], probe_ids, frozen_r["weights"]), frozen_r["weights"], frozen_r["r_readout"].orientation, True) for index, config in enumerate(configs)}
#     abstract, neural, scales = normalize_blocks(repeated_r_signature(data, "Dfit", 2), neural, data.pairs["Dfit"], 2)
#     selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
#     return {"selector": selector, "normalization_scales": scales}

def rank_against_frozen_r(ctx: Context, data: Data, candidates: Sequence[ChannelSite], frozen_r: Mapping[str, Any]) -> dict[str, Any]:
    probes = get_probe_sites(ctx, frozen_r["weights"])
    probe_ids = tuple(site.site_id for site in probes)
    configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidates)
    margins, values = run_probes(ctx, data, configs, "Dfit", probes)

    neural = {
        config.handle_id: neural_signature(
            data,
            "Dfit",
            margins[index],
            combine_probes(values[index], probe_ids, frozen_r["weights"]),
            frozen_r["weights"],
            frozen_r["r_readout"].orientation,
            False,
        )
        for index, config in enumerate(configs)
    }

    abstract = repeated_r_signature(data, "Dfit", 1)
    selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
    return {"selector": selector, "normalization_scales": {}}


def calibrate_against_frozen_r(ctx: Context, data: Data, ranked_pool: Sequence[Mapping[str, Any]], frozen_r: Mapping[str, Any], strengths: Sequence[float], run_heldout: bool = True) -> dict[str, Any]:
    probes = get_probe_sites(ctx, frozen_r["weights"])
    probe_ids = tuple(site.site_id for site in probes)
    configs, metadata = make_calibration_configs(ranked_pool, strengths)
    cal_margins, cal_probes = run_probes(ctx, data, configs, "Dcal", probes)
    downstream = combine_probes(cal_probes, probe_ids, frozen_r["weights"])
    grid = [{**row, "summary": r_summary(data, "Dcal", cal_margins[index], downstream[index], frozen_r["weights"], frozen_r["r_readout"])} for index, row in enumerate(metadata)]
    best, tier = get_best_tier(grid)
    calibrated = best_strength_per_handle(grid)
    if not calibrated:
        return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": [], "heldout": [], "accepted": []}
    if not run_heldout:
        return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": calibrated, "heldout": [], "accepted": []}

    test_configs = tuple(HandleConfiguration(row["handle_id"], row["weights"], row["strength"]) for row in calibrated)
    test_margins, test_probes = run_probes(ctx, data, test_configs, "Dte", probes)
    restored_margins, _ = run_probes(ctx, data, test_configs, "Dte", probes, probe_ids)
    downstream = combine_probes(test_probes, probe_ids, frozen_r["weights"])
    heldout = []
    for index, row in enumerate(calibrated):
        summary = r_summary(data, "Dte", test_margins[index], downstream[index], frozen_r["weights"], frozen_r["r_readout"])
        mediation = mediation_summary(data.pairs["Dte"], data.by_id, data.runs, test_margins[index], restored_margins[index])
        heldout.append({**row, "heldout": summary, "mediation": mediation, "accepted": is_certified(summary, mediation)})
    return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": calibrated, "heldout": heldout, "accepted": [row for row in heldout if row["accepted"]]}


def rank_against_frozen_d(ctx: Context, data: Data, candidates: Sequence[ChannelSite], frozen_d: Mapping[str, Any]) -> dict[str, Any]:
    d_weights = frozen_d["weights"]
    d_decoder, _ = fit_d(data.examples, data.runs, d_weights, SELECTION_SPLITS)
    probes = get_probe_sites(ctx, d_weights)
    probe_ids = tuple(site.site_id for site in probes)
    configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidates)
    margins, values = run_probes(ctx, data, configs, "Dfit", probes)
    neural = {config.handle_id: neural_signature(data, "Dfit", margins[index], combine_probes(values[index], probe_ids, d_weights), d_weights, d_decoder.slope, False) for index, config in enumerate(configs)}
    abstract = abstract_e_signature(data.pairs["Dfit"], data.by_id, definition="active_depth")
    abstract, neural, scales = normalize_blocks(abstract, neural, data.pairs["Dfit"], 1)
    selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
    return {"selector": selector, "normalization_scales": scales}


def test_against_frozen_d(ctx: Context, data: Data, rows: Sequence[Mapping[str, Any]], frozen_d: Mapping[str, Any], r_weights: Mapping[str, float], r_readout: Any) -> list[dict[str, Any]]:
    if not rows:
        return []
    d_weights = frozen_d["weights"]
    d_decoder, _ = fit_d(data.examples, data.runs, d_weights, SELECTION_SPLITS)
    probes = get_probe_sites(ctx, d_weights, r_weights)
    probe_ids = tuple(site.site_id for site in probes)
    configs = tuple(HandleConfiguration(row["handle_id"], row["weights"], row["strength"]) for row in rows)
    margins, values = run_probes(ctx, data, configs, "Dte", probes)
    restored_margins, _ = run_probes(ctx, data, configs, "Dte", probes, tuple(d_weights))
    heldout = []
    for index, row in enumerate(rows):
        summary = d_summary(data, "Dte", margins[index], values[index], probe_ids, d_weights, d_decoder, r_weights, r_readout)
        mediation = mediation_summary(data.pairs["Dte"], data.by_id, data.runs, margins[index], restored_margins[index])
        heldout.append({**row, "heldout": summary, "mediation": mediation, "accepted": is_certified(summary, mediation)})
    return heldout


def calibrate_against_frozen_d(ctx: Context, data: Data, ranked_pool: Sequence[Mapping[str, Any]], frozen_d: Mapping[str, Any], r_weights: Mapping[str, float], r_readout: Any, strengths: Sequence[float], run_heldout: bool = True) -> dict[str, Any]:
    d_weights = frozen_d["weights"]
    d_decoder, _ = fit_d(data.examples, data.runs, d_weights, SELECTION_SPLITS)
    probes = get_probe_sites(ctx, d_weights, r_weights)
    probe_ids = tuple(site.site_id for site in probes)
    configs, metadata = make_calibration_configs(ranked_pool, strengths)
    cal_margins, cal_probes = run_probes(ctx, data, configs, "Dcal", probes)
    grid = [{**row, "summary": d_summary(data, "Dcal", cal_margins[index], cal_probes[index], probe_ids, d_weights, d_decoder, r_weights, r_readout)} for index, row in enumerate(metadata)]
    best, tier = get_best_tier(grid)
    calibrated = best_strength_per_handle(grid)
    if not calibrated:
        return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": [], "heldout": [], "accepted": []}
    if not run_heldout:
        return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": calibrated, "heldout": [], "accepted": []}
    heldout = test_against_frozen_d(ctx, data, calibrated, frozen_d, r_weights, r_readout)
    return {"calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": calibrated, "heldout": heldout, "accepted": [row for row in heldout if row["accepted"]]}


def choose_r(handles: Sequence[Mapping[str, Any]], rank: Mapping[str, int]) -> Mapping[str, Any] | None:
    choices = [row for row in handles if not row["is_D"] and min(row["r_accuracy"].values()) >= 0.9]
    return min(choices, key=lambda row: (int(row["row"]["k"]), min(rank.get(site_id, 10**9) for site_id in row["weights"]))) if choices else None


def choose_earliest_r(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    choices = [row for row in handles if not row["is_D"] and min(row["r_accuracy"].values()) >= 0.9]
    return min(choices, key=lambda row: (handle_order(row["weights"]), int(row["row"]["k"]))) if choices else None


def choose_d(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    choices = [row for row in handles if row["is_D"]]
    return max(choices, key=lambda row: (float(row["row"]["summary"]["score"]), -int(row["row"]["k"]), -abs(float(row["row"]["strength"]) - 1.0), abs(float(row["d_metrics"]["Dcal"]["pearson"])))) if choices else None


# def choose_earliest_d(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
#     choices = [row for row in handles if row["is_D"]]
#     return min(choices, key=lambda row: (handle_order(row["weights"]), int(row["row"]["k"]))) if choices else None

def choose_earliest_d(handles):
    valid = [row for row in handles if row["is_D"]]
    if not valid:
        return None
    return min(valid, key=lambda row: (
        -float(row["row"]["summary"]["score"]),
        -abs(float(row["d_metrics"]["Dcal"]["pearson"])),
        float(row["d_metrics"]["Dcal"]["mae"]),
        handle_order(row["weights"]),
        int(row["row"]["k"]),
        abs(float(row["row"]["strength"]) - 1.0),
    ))


def label_chain(chain: list[dict[str, Any]], variable: str) -> None:
    if len(chain) == 1:
        chain[0]["name"] = variable
        return
    chain[0]["name"] = f"{variable}_late"
    chain[-1]["name"] = f"{variable}_early"
    for index in range(1, len(chain) - 1):
        chain[index]["name"] = f"{variable}_mid_{len(chain) - index - 1}"


def main() -> None:
    # Setup.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = make_tinypython_encoding(args.circuit_home)
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = convert_transformer_linears_to_sparse(model)
    ctx = Context(args, model, circuit.sites, {site.site_id: site for site in circuit.sites}, int(enc.encode("]\n")[0]), int(enc.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")
    strengths = parse_numbers(args.strength_grid)

    # Coarse X -> R -> Y.
    coarse_examples = build_bracket_rediscovery_bank(enc, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse_pairs = {split: build_bracket_pairs(coarse_examples, split=split, records_per_relation=100) for split in SPLITS}
    coarse = make_data(ctx, coarse_examples, coarse_pairs)
    coarse_result = coarse_phase(ctx, coarse, strengths)
    print("finish this phase: X -> R -> Y")

    # Independent graded bank used to distinguish binary R from graded D.
    graded_examples = build_graded_evidence_bank(enc, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    graded_limits = {"Dfit": 64, "Dcal": 48, "Dte": 64}
    graded_pairs = {split: build_graded_pairs(graded_examples, split=split, records_per_relation=graded_limits[split], e_definition="active_depth") for split in SPLITS}
    graded = make_data(ctx, graded_examples, graded_pairs)
    print("finish this phase: distinguish binary R from graded D.")

    # Select R_late from the coarse-certified handles.
    classified = classify_handles(coarse_result["accepted"], graded, args.graded_threshold)
    coarse_result["classified_handles"] = [public_handle(row) for row in classified]
    rank = {str(row["site_id"]): index for index, row in enumerate(coarse_result["selector"]["ranked"])}
    r_current = choose_r(classified, rank)
    if r_current is None:
        raise RuntimeError("no binary-only R handle passed coarse validation")
    r_chain = [{"name": "R", **public_handle(r_current), "source": "coarse"}]
    print("finish this phase: Select R_late from the coarse-certified handles.")

    # Search for R_early only inside the earlier part of the coarse R candidate pool.
    coarse_pool = tuple(ctx.site_lookup[str(row["site_id"])] for row in coarse_result["effective_support"])
    earlier_r_sites = find_earlier_sites(coarse_pool, r_current["weights"])
    r_refinement = None
    if earlier_r_sites:
        print(f"[2] Earliest R search: {len(earlier_r_sites)} candidates", flush=True)
        r_ranking = rank_against_frozen_r(ctx, coarse, earlier_r_sites, r_current)
        r_evaluation = calibrate_against_frozen_r(ctx, coarse, r_ranking["selector"]["ranked"], r_current, strengths)
        r_classified = classify_handles(r_evaluation["accepted"], graded, args.graded_threshold)
        r_candidate = choose_earliest_r(r_classified)
        r_refinement = {"candidate_ids": [site.site_id for site in earlier_r_sites], **r_ranking, **r_evaluation, "classified_handles": [public_handle(row) for row in r_classified]}
        atomic_json(args.out_dir / "R_refinement.json", r_refinement)
        if r_candidate is not None:
            r_current = r_candidate
            r_chain.append({"name": "R_early", **public_handle(r_current), "source": "earliest certified handle in the coarse R candidate pool"})
    print("finish this phase: Search for R_early only inside the earlier part of the coarse R candidate pool.")

    # Start a fresh D search after the R chain is finished.
    d_sites = find_earlier_sites(ctx.sites, r_current["weights"])
    if not d_sites:
        raise RuntimeError("no site is earlier than the final R handle")
    print(f"[3] D discovery: {len(d_sites)} upstream candidates", flush=True)
    d_ranking = rank_against_frozen_r(ctx, graded, d_sites, r_current)
    d_support = get_support(d_ranking["selector"]["ranked"], args.top_n, args.mass_fraction)
    d_evaluation = calibrate_against_frozen_r(ctx, graded, d_support, r_current, strengths, run_heldout=False)
    d_classified = classify_handles(d_evaluation["calibrated_handles"], graded, args.graded_threshold)
    d_current = choose_d(d_classified)
    if d_current is None:
        raise RuntimeError("no graded D handle passed the independent upstream D search")
    d_discovery = {"candidate_ids": [site.site_id for site in d_sites], **d_ranking, "effective_support": d_support, **d_evaluation, "classified_handles": [public_handle(row) for row in d_classified]}
    atomic_json(args.out_dir / "D_discovery.json", d_discovery)
    print("finish this phase: Start a fresh D search after the R chain is finished.")

    # Directly validate D -> R -> Y.
    d_weights = d_current["weights"]
    r_weights = r_current["weights"]
    d_decoder, d_metrics = fit_d(graded.examples, graded.runs, d_weights)
    r_readout, r_accuracy = fit_r(graded.examples, graded.runs, r_weights)
    direct_config = HandleConfiguration("direct_D", d_weights, float(d_current["row"]["strength"]))
    direct_probes = get_probe_sites(ctx, d_weights, r_weights)
    direct_probe_ids = tuple(site.site_id for site in direct_probes)
    direct_splits = {}
    for split in ("Dcal", "Dte"):
        margins, values = run_probes(ctx, graded, (direct_config,), split, direct_probes)
        restored_margins, _ = run_probes(ctx, graded, (direct_config,), split, direct_probes, tuple(r_weights))
        summary = d_summary(graded, split, margins[0], values[0], direct_probe_ids, d_weights, d_decoder, r_weights, r_readout)
        mediation = mediation_summary(graded.pairs[split], graded.by_id, graded.runs, margins[0], restored_margins[0])
        direct_splits[split] = {"summary": summary, "R_mediation": mediation, "accepted": is_certified(summary, mediation)}
    if not all(row["accepted"] for row in direct_splits.values()):
        raise RuntimeError("the Dfit/Dcal-selected D handle failed direct D -> R -> Y validation")
    direct_d = {"weights": d_weights, "decoder": d_decoder.to_dict(), "metrics": d_metrics, "R_readout": r_readout.to_dict(), "R_accuracy": r_accuracy, "splits": direct_splits}
    d_current = {**d_current, "d_decoder": d_decoder, "d_metrics": d_metrics, "r_readout": r_readout, "r_accuracy": r_accuracy}
    d_chain = [{"name": "D", **public_handle(d_current), "source": "independent D discovery"}]
    print("finish this phase: Directly validate D -> R -> Y.")

    # Search once for the earliest certified D inside the previous D candidate pool.
    d_pool = tuple(ctx.site_lookup[str(row["site_id"])] for row in d_support)
    earlier_d_sites = tuple(site for site in find_earlier_sites(d_pool, d_current["weights"]) if site.site_id not in r_weights)
    d_refinement = None
    if earlier_d_sites:
        print(f"[4] Earliest D search: {len(earlier_d_sites)} candidates", flush=True)
        stage_ranking = rank_against_frozen_d(ctx, graded, earlier_d_sites, d_current)
        stage_evaluation = calibrate_against_frozen_d(ctx, graded, stage_ranking["selector"]["ranked"], d_current, r_weights, r_readout, strengths, run_heldout=False)
        stage_classified = classify_handles(stage_evaluation["calibrated_handles"], graded, args.graded_threshold)
        candidate = choose_earliest_d(stage_classified)
        selected_heldout = test_against_frozen_d(ctx, graded, (candidate["row"],), d_current, r_weights, r_readout) if candidate is not None else []
        if not selected_heldout or not selected_heldout[0]["accepted"]:
            candidate = None
        stage_evaluation["heldout"] = selected_heldout
        stage_evaluation["accepted"] = [row for row in selected_heldout if row["accepted"]]
        d_refinement = {"candidate_ids": [site.site_id for site in earlier_d_sites], **stage_ranking, **stage_evaluation, "classified_handles": [public_handle(row) for row in stage_classified]}
        atomic_json(args.out_dir / "D_refinement.json", d_refinement)
        if candidate is not None:
            candidate_decoder, candidate_metrics = fit_d(graded.examples, graded.runs, candidate["weights"])
            candidate_readout, candidate_accuracy = fit_r(graded.examples, graded.runs, candidate["weights"])
            candidate = {**candidate, "d_decoder": candidate_decoder, "d_metrics": candidate_metrics, "r_readout": candidate_readout, "r_accuracy": candidate_accuracy}
            d_current = candidate
            d_chain.append({"name": "D_early", **public_handle(d_current), "source": "earliest certified handle in the D candidate pool"})

    label_chain(r_chain, "R")
    label_chain(d_chain, "D")
    result = {
        "experiment": "automatic_gradual_PLOT_bracket",
        "declared_model": "X -> D -> R -> Y, D=active bracket depth, R=1[D>=2]",
        "scope": "The code localizes declared variables; it does not invent new causal variables.",
        "rules": {"support": f"top {args.top_n}, mass >= {args.mass_fraction:g} * top-1", "calibration": "all singletons and pairs", "D_selection": "Dcal score, smaller K, strength closest to 1, then Dcal depth Pearson", "heldout": "Dte is evaluated only after the D handle is selected and frozen"},
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "banks": {"coarse": bank_manifest(coarse.examples, coarse.pairs), "graded": bank_manifest(graded.examples, graded.pairs)},
        "clean_accuracy": {"coarse": {split: clean_accuracy(coarse.examples, coarse.runs, split=split) for split in SPLITS}, "graded": {split: clean_accuracy(graded.examples, graded.runs, split=split) for split in SPLITS}},
        "coarse": {key: value for key, value in coarse_result.items() if key != "accepted"},
        "R_refinement": r_refinement,
        "R_chain": r_chain,
        "D_discovery": d_discovery,
        "direct_D_to_R_to_Y": direct_d,
        "D_refinement": d_refinement,
        "D_chain": d_chain,
        "final_model": "X -> " + " -> ".join([row["name"] for row in reversed(d_chain)] + [row["name"] for row in reversed(r_chain)]) + " -> Y",
    }
    output = args.out_dir / "automatic_gradual_discovery.json"
    atomic_json(output, result)
    print(json.dumps({"status": "complete", "final_model": result["final_model"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()