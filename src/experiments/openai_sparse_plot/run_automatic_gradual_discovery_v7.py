


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
    ClampedRun, HandleConfiguration, RediscoveryExample, RediscoveryPair, abstract_signature, atomic_json, bank_manifest,
    build_bracket_pairs, build_bracket_rediscovery_bank, clean_accuracy, collect_clamped_runs, evaluate_configurations,
    load_candidate_circuit, match_signatures, relation_summary, select_calibration_row,
)
from .activation import ChannelSite
from .bracket_progressive_model_discovery import layer_order
from .graded_evidence import abstract_e_signature as abstract_d_signature, build_graded_evidence_bank as build_graded_d_bank, build_graded_pairs, decoder_metrics, e_value as d_value, fit_affine_decoder, graded_validation_summary
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
    runs: dict[str, ClampedRun]


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
    # Read experiment settings from the command line.
    parser = argparse.ArgumentParser(description="Automatic gradual PLOT for X -> D -> R -> Y.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/automatic_gradual_discovery_v9"))
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def run_output(ctx: Context, data: Data, configs: Sequence[HandleConfiguration], split: str) -> np.ndarray:
    # Intervene with each configuration and return its output margins.
    return evaluate_configurations(ctx.model, configs, data.pairs[split], examples=data.by_id, runs=data.runs, site_lookup=ctx.site_lookup, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)


def run_and_measure_downstream(ctx: Context, data: Data, configs: Sequence[HandleConfiguration], split: str, downstream_sites: Sequence[ChannelSite], restore_downstream_sites: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray]:
    # Intervene on source candidates and measure Y plus the frozen downstream sites.
    return evaluate_progressive_configurations(ctx.model, configs, data.pairs[split], examples=data.by_id, runs=data.runs, site_lookup=ctx.site_lookup, probe_sites=downstream_sites, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size, restore_probe_site_ids=restore_downstream_sites)


def make_data(ctx: Context, examples: tuple[RediscoveryExample, ...], pairs: dict[str, tuple[RediscoveryPair, ...]]) -> Data:
    # Collect model activations for Dfit and Dcal only; do not load Dte yet.
    selection_examples = tuple(row for row in examples if row.split in SELECTION_SPLITS)
    runs = collect_clamped_runs(ctx.model, selection_examples, candidate_sites=ctx.sites, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)
    return Data(examples, pairs, {row.example_id: row for row in examples}, dict(runs))


def load_dte(ctx: Context, data: Data) -> None:
    # Load Dte activations after all handles and strengths have been frozen.
    examples = tuple(row for row in data.examples if row.split == "Dte")
    runs = collect_clamped_runs(ctx.model, examples, candidate_sites=ctx.sites, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)
    data.runs.update(runs)


# -------------------- Common handle utilities --------------------

def normalize_blocks(abstract: Sequence[float], neural: Mapping[str, Sequence[float]], pairs: Sequence[RediscoveryPair], components: int) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], dict[str, float]]:
    # Put abstract and neural signatures on the same scale for OT ranking.
    abstract = torch.tensor(abstract, dtype=torch.float32)
    site_ids = tuple(neural)
    neural_tensor = torch.tensor([neural[site_id] for site_id in site_ids], dtype=torch.float32)
    scales = {}
    for relation in sorted({pair.relation for pair in pairs}):
        for component in range(components):
            indices = [components * i + component for i, pair in enumerate(pairs) if pair.relation == relation]
            scale = max(float(torch.sqrt((abstract[indices] ** 2).mean())), float(torch.sqrt((neural_tensor[:, indices] ** 2).mean())), 1e-6)
            abstract[indices] /= scale
            neural_tensor[:, indices] /= scale
            scales[f"{relation}:{component}"] = scale
    neural = {site_id: tuple(float(value) for value in neural_tensor[i]) for i, site_id in enumerate(site_ids)}
    return tuple(float(value) for value in abstract), neural, scales


def get_support(ranked: Sequence[Mapping[str, Any]], top_n: int, mass_fraction: float) -> list[dict[str, Any]]:
    # Keep the top OT-ranked sites that also pass the OT-mass cutoff.
    support = [dict(row) for row in ranked[:top_n]]
    if not support:
        return []
    cutoff = mass_fraction * float(support[0]["weight"])
    return [row for row in support if float(row["weight"]) >= cutoff]


def make_calibration_configs(support: Sequence[Mapping[str, Any]], strengths: Sequence[float]) -> tuple[tuple[HandleConfiguration, ...], list[dict[str, Any]]]:
    # Build every singleton/pair handle at every intervention strength.
    configs, metadata = [], []
    for k in (1, 2):
        for index, sites in enumerate(combinations(support, k), start=1):
            total = sum(float(site["weight"]) for site in sites)
            weights = {str(site["site_id"]): float(site["weight"]) / total for site in sites}
            for strength in strengths:
                handle_id = f"S{k}_{index}_lambda{strength:g}"
                configs.append(HandleConfiguration(handle_id, weights, strength))
                metadata.append({"handle_id": handle_id, "k": k, "strength": strength, "weights": weights, "site_ids": list(weights)})
    return tuple(configs), metadata


def best_tier(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Return the best Dcal row and every row tied at the best causal score.
    score = max(float(row["summary"]["score"]) for row in rows)
    tier = [dict(row) for row in rows if abs(float(row["summary"]["score"]) - score) < 1e-12]
    return dict(select_calibration_row(tier)), tier


def best_strength_per_handle(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Keep the best passing Dcal strength for each singleton/pair support.
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["site_ids"])].append(row)
    selected = []
    for group in groups.values():
        row = min(group, key=lambda x: (-float(x["summary"]["score"]), abs(float(x["strength"]) - 1.0)))
        if row["summary"]["passes"]:
            selected.append(dict(row))
    return selected


def handle_value(run: ClampedRun, weights: Mapping[str, float]) -> float:
    # Compute one handle value as the weighted sum of its component sites.
    return sum(float(weight) * float(run.features_by_site[site_id]) for site_id, weight in weights.items())


def handle_order(weights: Mapping[str, float]) -> tuple[int, int]:
    # Use the latest component site as the position of a handle.
    return max(layer_order(site_id) for site_id in weights)


def earlier_sites(sites: Sequence[ChannelSite], downstream: Mapping[str, float]) -> tuple[ChannelSite, ...]:
    # Keep sites located before a frozen downstream handle.
    cutoff = handle_order(downstream)
    return tuple(site for site in sites if site.site_id not in downstream and layer_order(site.site_id) < cutoff)


def get_downstream_sites(ctx: Context, *handles: Mapping[str, float]) -> tuple[ChannelSite, ...]:
    # Get the component sites of the frozen handles that we need to measure.
    site_ids = []
    for handle in handles:
        for site_id in handle:
            if site_id not in site_ids:
                site_ids.append(site_id)
    return tuple(ctx.site_lookup[site_id] for site_id in site_ids)


def combine_downstream_values(downstream_values: np.ndarray, downstream_site_ids: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    # Combine measured component-site values into one downstream handle value.
    index = {site_id: i for i, site_id in enumerate(downstream_site_ids)}
    return sum(float(weight) * downstream_values[..., index[site_id]] for site_id, weight in weights.items())


def fit_r(data: Data, weights: Mapping[str, float], splits: Sequence[str] = SELECTION_SPLITS) -> tuple[Any, dict[str, float]]:
    # Fit a binary R readout on Dfit and measure its accuracy on each split.
    fit_rows = [row for row in data.examples if row.split == "Dfit"]
    readout = fit_binary_scalar_readout([handle_value(data.runs[row.example_id], weights) for row in fit_rows], [row.variable_value for row in fit_rows])
    accuracy = {split: float(np.mean([readout.predict(handle_value(data.runs[row.example_id], weights)) == row.variable_value for row in data.examples if row.split == split])) for split in splits}
    return readout, accuracy


def fit_d(data: Data, weights: Mapping[str, float], splits: Sequence[str] = SELECTION_SPLITS) -> tuple[Any, dict[str, dict[str, float]]]:
    # Fit a graded D decoder on Dfit and measure Pearson/MAE on each split.
    fit_rows = [row for row in data.examples if row.split == "Dfit"]
    decoder = fit_affine_decoder([handle_value(data.runs[row.example_id], weights) for row in fit_rows], [d_value(row, "active_depth") for row in fit_rows])
    metrics = {}
    for split in splits:
        rows = [row for row in data.examples if row.split == split]
        metrics[split] = decoder_metrics(decoder, [handle_value(data.runs[row.example_id], weights) for row in rows], [d_value(row, "active_depth") for row in rows])
    return decoder, metrics


def classify_handles(rows: Sequence[Mapping[str, Any]], graded: Data, threshold: float) -> list[dict[str, Any]]:
    # Fit both readouts and classify each calibrated handle as binary R or graded D.
    handles, seen = [], set()
    for row in rows:
        weights = {str(site_id): float(weight) for site_id, weight in row["weights"].items()}
        key = tuple(sorted(weights.items()))
        if key in seen:
            continue
        seen.add(key)
        r_readout, r_accuracy = fit_r(graded, weights)
        d_decoder, d_metrics = fit_d(graded, weights)
        is_d = min(abs(float(d_metrics[split]["pearson"])) for split in SELECTION_SPLITS) >= threshold
        handles.append({"row": dict(row), "weights": weights, "r_readout": r_readout, "r_accuracy": r_accuracy, "d_decoder": d_decoder, "d_metrics": d_metrics, "is_D": is_d})
    return handles


def public_handle(handle: Mapping[str, Any]) -> dict[str, Any]:
    # Remove Python readout objects so the selected handle can be saved as JSON.
    return {"handle_id": handle["row"]["handle_id"], "weights": handle["weights"], "strength": handle["row"]["strength"], "R_accuracy": handle["r_accuracy"], "D_decoder": handle["d_decoder"].to_dict(), "D_metrics": handle["d_metrics"], "is_D": handle["is_D"]}


# -------------------- Causal summaries --------------------

def expected_r(pair: RediscoveryPair, examples: Mapping[str, RediscoveryExample]) -> int:
    # Return the R value expected after patching source into base.
    base, source = examples[pair.base_id].variable_value, examples[pair.source_id].variable_value
    return int(source if source != base else base)


def r_summary(data: Data, split: str, margins: Sequence[float], downstream: Sequence[float], frozen_r: Mapping[str, Any]) -> dict[str, Any]:
    # Measure sensitivity and invariance for an upstream variable controlling R.
    by_relation, sensitivity, invariance = defaultdict(list), [], []
    for i, pair in enumerate(data.pairs[split]):
        base = handle_value(data.runs[pair.base_id], frozen_r["weights"])
        source = handle_value(data.runs[pair.source_id], frozen_r["weights"])
        patched, target_r = float(downstream[i]), expected_r(pair, data.by_id)
        row = {"output_correct": (1 if float(margins[i]) > 0 else -1) == target_r, "downstream_correct": frozen_r["r_readout"].predict(patched) == target_r, "downstream_moves": abs(source - patched) < abs(source - base) if abs(source - base) > 1e-8 else abs(patched - base) <= 1e-6}
        by_relation[pair.relation].append(row)
        (sensitivity if data.by_id[pair.source_id].variable_value != data.by_id[pair.base_id].variable_value else invariance).append(row)
    relations = {relation: {key: float(np.mean([row[key] for row in rows])) for key in ("output_correct", "downstream_correct", "downstream_moves")} for relation, rows in sorted(by_relation.items())}
    blocks = {"sensitivity_output": float(np.mean([row["output_correct"] for row in sensitivity])), "sensitivity_downstream": float(np.mean([row["downstream_correct"] for row in sensitivity])), "invariance_output": float(np.mean([row["output_correct"] for row in invariance])), "invariance_downstream": float(np.mean([row["downstream_correct"] for row in invariance]))}
    return {"relations": relations, "balanced_blocks": blocks, "score": float(np.mean(list(blocks.values()))), "passes": min(blocks.values()) >= 0.9}


def d_summary(data: Data, split: str, margins: Sequence[float], downstream_values: np.ndarray, downstream_site_ids: Sequence[str], frozen_d: Mapping[str, Any], frozen_r: Mapping[str, Any]) -> dict[str, Any]:
    # Measure whether an intervention changes frozen D, R, and Y correctly.
    abstract_d = {example_id: d_value(data.by_id[example_id], "active_depth") for example_id in data.runs}
    clean_d = {example_id: frozen_d["d_decoder"].predict(handle_value(run, frozen_d["weights"])) for example_id, run in data.runs.items()}
    patched_d = frozen_d["d_decoder"].slope * combine_downstream_values(downstream_values, downstream_site_ids, frozen_d["weights"]) + frozen_d["d_decoder"].intercept
    patched_r = [frozen_r["r_readout"].predict(value) for value in combine_downstream_values(downstream_values, downstream_site_ids, frozen_r["weights"])]
    patched_y = np.where(np.asarray(margins) > 0, 1, -1)
    return graded_validation_summary(data.pairs[split], data.by_id, abstract_d, clean_d, patched_d, patched_r, patched_y)


def is_certified(summary: Mapping[str, Any], mediation: Mapping[str, Any]) -> bool:
    # Pass an edge only when its causal behavior and restoration test both pass.
    return bool(summary["passes"] and mediation["passes"])


# -------------------- Ranking and calibration: Dfit/Dcal only --------------------

def coarse_phase(ctx: Context, data: Data, strengths: Sequence[float]) -> dict[str, Any]:
    # Rank all 133 sites on Dfit, then calibrate singleton/pair R handles on Dcal.
    print("[1] Coarse R search", flush=True)
    configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in ctx.sites)
    margins = run_output(ctx, data, configs, "Dfit")
    neural = {config.handle_id: tuple(float(margins[i, j] - data.runs[pair.base_id].class_margin) for j, pair in enumerate(data.pairs["Dfit"])) for i, config in enumerate(configs)}
    abstract, neural, scales = normalize_blocks(abstract_signature(data.pairs["Dfit"], data.by_id), neural, data.pairs["Dfit"], 1)
    selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
    support = get_support(selector["ranked"], ctx.args.top_n, ctx.args.mass_fraction)

    configs, metadata = make_calibration_configs(support, strengths)
    margins = run_output(ctx, data, configs, "Dcal")
    grid = [{**row, "summary": relation_summary(data.pairs["Dcal"], data.by_id, margins[i])} for i, row in enumerate(metadata)]
    best, tier = best_tier(grid)
    return {"selector": selector, "normalization_scales": scales, "effective_support": support, "calibration_grid": grid, "calibration_best": best, "calibration_best_tier": tier, "heldout": []}


def neural_signature(data: Data, split: str, downstream: Sequence[float], frozen_weights: Mapping[str, float], scale: float) -> tuple[float, ...]:
    # Record how much a source intervention moves the frozen downstream handle.
    return tuple(scale * (float(downstream[i]) - handle_value(data.runs[pair.base_id], frozen_weights)) for i, pair in enumerate(data.pairs[split]))


def rank_against_frozen_r(ctx: Context, data: Data, candidates: Sequence[ChannelSite], frozen_r: Mapping[str, Any]) -> dict[str, Any]:
    # Rank source candidates by how correctly they move the frozen R on Dfit.
    downstream_sites = get_downstream_sites(ctx, frozen_r["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidates)
    _, downstream_values = run_and_measure_downstream(ctx, data, configs, "Dfit", downstream_sites)
    neural = {config.handle_id: neural_signature(data, "Dfit", combine_downstream_values(downstream_values[i], downstream_site_ids, frozen_r["weights"]), frozen_r["weights"], frozen_r["r_readout"].orientation) for i, config in enumerate(configs)}
    abstract = tuple(float(data.by_id[pair.source_id].variable_value - data.by_id[pair.base_id].variable_value) for pair in data.pairs["Dfit"])
    return {"selector": match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta), "normalization_scales": {}}


def calibrate_against_frozen_r(ctx: Context, data: Data, ranked: Sequence[Mapping[str, Any]], frozen_r: Mapping[str, Any], strengths: Sequence[float]) -> dict[str, Any]:
    # Test candidate singleton/pair handles for controlling frozen R on Dcal.
    downstream_sites = get_downstream_sites(ctx, frozen_r["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    configs, metadata = make_calibration_configs(ranked, strengths)
    margins, downstream_values = run_and_measure_downstream(ctx, data, configs, "Dcal", downstream_sites)
    downstream = combine_downstream_values(downstream_values, downstream_site_ids, frozen_r["weights"])
    grid = [{**row, "summary": r_summary(data, "Dcal", margins[i], downstream[i], frozen_r)} for i, row in enumerate(metadata)]
    best, tier = best_tier(grid)
    return {"calibration_grid": grid, "calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": best_strength_per_handle(grid), "heldout": []}


def rank_against_frozen_d(ctx: Context, data: Data, candidates: Sequence[ChannelSite], frozen_d: Mapping[str, Any]) -> dict[str, Any]:
    # Rank source candidates by how correctly they move the frozen D on Dfit.
    downstream_sites = get_downstream_sites(ctx, frozen_d["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidates)
    _, downstream_values = run_and_measure_downstream(ctx, data, configs, "Dfit", downstream_sites)
    neural = {config.handle_id: neural_signature(data, "Dfit", combine_downstream_values(downstream_values[i], downstream_site_ids, frozen_d["weights"]), frozen_d["weights"], frozen_d["d_decoder"].slope) for i, config in enumerate(configs)}
    abstract = abstract_d_signature(data.pairs["Dfit"], data.by_id, definition="active_depth")
    abstract, neural, scales = normalize_blocks(abstract, neural, data.pairs["Dfit"], 1)
    return {"selector": match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta), "normalization_scales": scales}


def calibrate_against_frozen_d(ctx: Context, data: Data, ranked: Sequence[Mapping[str, Any]], frozen_d: Mapping[str, Any], frozen_r: Mapping[str, Any], strengths: Sequence[float]) -> dict[str, Any]:
    # Test candidate singleton/pair handles for controlling frozen D on Dcal.
    downstream_sites = get_downstream_sites(ctx, frozen_d["weights"], frozen_r["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    configs, metadata = make_calibration_configs(ranked, strengths)
    margins, downstream_values = run_and_measure_downstream(ctx, data, configs, "Dcal", downstream_sites)
    grid = [{**row, "summary": d_summary(data, "Dcal", margins[i], downstream_values[i], downstream_site_ids, frozen_d, frozen_r)} for i, row in enumerate(metadata)]
    best, tier = best_tier(grid)
    return {"calibration_grid": grid, "calibration_best": best, "calibration_best_tier": tier, "calibrated_handles": best_strength_per_handle(grid), "heldout": []}


# -------------------- Selection rules --------------------

def choose_r(handles: Sequence[Mapping[str, Any]], site_rank: Mapping[str, int]) -> Mapping[str, Any] | None:
    # Choose a binary R with >=90% accuracy; prefer smaller K then better OT rank.
    valid = [row for row in handles if not row["is_D"] and min(row["r_accuracy"].values()) >= 0.9]
    return min(valid, key=lambda row: (int(row["row"]["k"]), min(site_rank.get(site_id, 10**9) for site_id in row["weights"]))) if valid else None


def choose_early_r(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    # Choose a valid upstream R; prefer the earlier handle then smaller K.
    valid = [row for row in handles if not row["is_D"] and min(row["r_accuracy"].values()) >= 0.9]
    return min(valid, key=lambda row: (handle_order(row["weights"]), int(row["row"]["k"]))) if valid else None


def choose_d(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    # Choose a graded D; prefer Dcal causal score, smaller K, strength, then Pearson.
    valid = [row for row in handles if row["is_D"]]
    return max(valid, key=lambda row: (float(row["row"]["summary"]["score"]), -int(row["row"]["k"]), -abs(float(row["row"]["strength"]) - 1.0), abs(float(row["d_metrics"]["Dcal"]["pearson"])))) if valid else None


def choose_early_d(handles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    # Choose an upstream D by causal score, Pearson, MAE, position, K, and strength.
    valid = [row for row in handles if row["is_D"]]
    return min(valid, key=lambda row: (-float(row["row"]["summary"]["score"]), -abs(float(row["d_metrics"]["Dcal"]["pearson"])), float(row["d_metrics"]["Dcal"]["mae"]), handle_order(row["weights"]), int(row["row"]["k"]), abs(float(row["row"]["strength"]) - 1.0))) if valid else None


def label_chain(chain: list[dict[str, Any]], variable: str) -> None:
    # Rename a one-handle chain as D/R or a two-handle chain as early/late.
    if len(chain) == 1:
        chain[0]["name"] = variable
    else:
        chain[0]["name"], chain[-1]["name"] = f"{variable}_late", f"{variable}_early"


# -------------------- Dte certification: called only at the end --------------------

def test_r_to_y(ctx: Context, data: Data, r_late: Mapping[str, Any]) -> dict[str, Any]:
    # Certify the selected R_late -> Y intervention on Dte.
    row = r_late["row"]
    margins = run_output(ctx, data, (HandleConfiguration(row["handle_id"], row["weights"], row["strength"]),), "Dte")[0]
    summary = relation_summary(data.pairs["Dte"], data.by_id, margins)
    return {**row, "heldout": summary, "accepted": bool(summary["all_rates_at_least_0_90"])}


def test_r_edge(ctx: Context, data: Data, early_r: Mapping[str, Any], late_r: Mapping[str, Any]) -> dict[str, Any]:
    # Certify R_early -> R_late -> Y and restoration on Dte.
    row, downstream_sites = early_r["row"], get_downstream_sites(ctx, late_r["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    config = HandleConfiguration(row["handle_id"], row["weights"], row["strength"])
    margins, downstream_values = run_and_measure_downstream(ctx, data, (config,), "Dte", downstream_sites)
    restored, _ = run_and_measure_downstream(ctx, data, (config,), "Dte", downstream_sites, downstream_site_ids)
    summary = r_summary(data, "Dte", margins[0], combine_downstream_values(downstream_values[0], downstream_site_ids, late_r["weights"]), late_r)
    mediation = mediation_summary(data.pairs["Dte"], data.by_id, data.runs, margins[0], restored[0])
    return {**row, "heldout": summary, "mediation": mediation, "accepted": is_certified(summary, mediation)}


def test_d_edge(ctx: Context, data: Data, source: Mapping[str, Any], frozen_d: Mapping[str, Any], frozen_r: Mapping[str, Any], restore_weights: Mapping[str, float]) -> dict[str, Any]:
    # Certify a source -> frozen D -> frozen R -> Y edge and restoration on Dte.
    row, downstream_sites = source["row"], get_downstream_sites(ctx, frozen_d["weights"], frozen_r["weights"])
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    config = HandleConfiguration(row["handle_id"], row["weights"], row["strength"])
    margins, downstream_values = run_and_measure_downstream(ctx, data, (config,), "Dte", downstream_sites)
    restored, _ = run_and_measure_downstream(ctx, data, (config,), "Dte", downstream_sites, tuple(restore_weights))
    summary = d_summary(data, "Dte", margins[0], downstream_values[0], downstream_site_ids, frozen_d, frozen_r)
    mediation = mediation_summary(data.pairs["Dte"], data.by_id, data.runs, margins[0], restored[0])
    return {**row, "heldout": summary, "mediation": mediation, "accepted": is_certified(summary, mediation)}


def short_edge(edge: Mapping[str, Any] | None) -> dict[str, Any] | None:
    # Keep only the main PASS/FAIL values for the readable summary file.
    if edge is None:
        return None
    summary = edge.get("heldout", {})
    mediation = edge.get("mediation")
    return {"passed": bool(edge["accepted"]), "causal_score": summary.get("score"), "restoration_passed": None if mediation is None else bool(mediation["passes"])}


def main() -> None:
    # Run discovery on Dfit/Dcal, freeze the model, then certify it on Dte.
    # Setup and datasets. make_data loads Dfit/Dcal only.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = make_tinypython_encoding(args.circuit_home)
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = convert_transformer_linears_to_sparse(model)
    ctx = Context(args, model, circuit.sites, {site.site_id: site for site in circuit.sites}, int(enc.encode("]\n")[0]), int(enc.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")
    strengths = tuple(float(value) for value in args.strength_grid.split(","))

    coarse_examples = build_bracket_rediscovery_bank(enc, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse = make_data(ctx, coarse_examples, {split: build_bracket_pairs(coarse_examples, split=split, records_per_relation=100) for split in SPLITS})
    graded_examples = build_graded_d_bank(enc, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    limits = {"Dfit": 64, "Dcal": 48, "Dte": 64}
    graded = make_data(ctx, graded_examples, {split: build_graded_pairs(graded_examples, split=split, records_per_relation=limits[split], e_definition="active_depth") for split in SPLITS})

    # Phase 1: find R_late on Dfit/Dcal.
    coarse_result = coarse_phase(ctx, coarse, strengths)
    classified = classify_handles(coarse_result["calibration_best_tier"], graded, args.graded_threshold)
    coarse_result["classified_handles"] = [public_handle(row) for row in classified]
    site_rank = {str(row["site_id"]): i for i, row in enumerate(coarse_result["selector"]["ranked"])}
    r_late = choose_r(classified, site_rank)
    if r_late is None:
        raise RuntimeError("No R handle passed Dfit/Dcal")

    # Phase 2: freeze R_late and look for R_early on Dfit/Dcal.
    r_early, r_refinement = None, None
    coarse_pool = tuple(ctx.site_lookup[str(row["site_id"])] for row in coarse_result["effective_support"])
    r_sites = earlier_sites(coarse_pool, r_late["weights"])
    if r_sites:
        print(f"[2] R refinement: {len(r_sites)} candidates", flush=True)
        ranking = rank_against_frozen_r(ctx, coarse, r_sites, r_late)
        calibration = calibrate_against_frozen_r(ctx, coarse, ranking["selector"]["ranked"], r_late, strengths)
        r_candidates = classify_handles(calibration["calibrated_handles"], graded, args.graded_threshold)
        r_early = choose_early_r(r_candidates)
        r_refinement = {"candidate_ids": [site.site_id for site in r_sites], **ranking, **calibration, "classified_handles": [public_handle(row) for row in r_candidates]}
    final_r = r_early or r_late
    r_chain = [{"name": "R", **public_handle(r_late)}] + ([{"name": "R_early", **public_handle(r_early)}] if r_early else [])
    label_chain(r_chain, "R")

    # Phase 3: freeze final R and find D_late on Dfit/Dcal.
    d_sites = earlier_sites(ctx.sites, final_r["weights"])
    print(f"[3] D discovery: {len(d_sites)} candidates", flush=True)
    d_ranking = rank_against_frozen_r(ctx, graded, d_sites, final_r)
    d_support = get_support(d_ranking["selector"]["ranked"], args.top_n, args.mass_fraction)
    d_calibration = calibrate_against_frozen_r(ctx, graded, d_support, final_r, strengths)
    d_candidates = classify_handles(d_calibration["calibrated_handles"], graded, args.graded_threshold)
    d_late = choose_d(d_candidates)
    if d_late is None:
        raise RuntimeError("No D handle passed Dfit/Dcal")
    d_discovery = {"candidate_ids": [site.site_id for site in d_sites], **d_ranking, "effective_support": d_support, **d_calibration, "classified_handles": [public_handle(row) for row in d_candidates]}

    # Validate D_late -> R -> Y on Dcal before refinement.
    dcal_downstream_sites = get_downstream_sites(ctx, d_late["weights"], final_r["weights"])
    dcal_downstream_site_ids = tuple(site.site_id for site in dcal_downstream_sites)
    dcal_config = HandleConfiguration(d_late["row"]["handle_id"], d_late["weights"], d_late["row"]["strength"])
    margins, downstream_values = run_and_measure_downstream(ctx, graded, (dcal_config,), "Dcal", dcal_downstream_sites)
    restored, _ = run_and_measure_downstream(ctx, graded, (dcal_config,), "Dcal", dcal_downstream_sites, tuple(final_r["weights"]))
    dcal_summary = d_summary(graded, "Dcal", margins[0], downstream_values[0], dcal_downstream_site_ids, d_late, final_r)
    dcal_mediation = mediation_summary(graded.pairs["Dcal"], graded.by_id, graded.runs, margins[0], restored[0])
    direct_d = {"weights": d_late["weights"], "decoder": d_late["d_decoder"].to_dict(), "metrics": d_late["d_metrics"], "R_readout": final_r["r_readout"].to_dict(), "R_accuracy": final_r["r_accuracy"], "splits": {"Dcal": {"summary": dcal_summary, "R_mediation": dcal_mediation, "accepted": is_certified(dcal_summary, dcal_mediation)}}}

    # Phase 4: freeze D_late and look for D_early on Dfit/Dcal.
    d_early, d_refinement = None, None
    d_pool = tuple(ctx.site_lookup[str(row["site_id"])] for row in d_support)
    early_d_sites = tuple(site for site in earlier_sites(d_pool, d_late["weights"]) if site.site_id not in final_r["weights"])
    if early_d_sites:
        print(f"[4] D refinement: {len(early_d_sites)} candidates", flush=True)
        ranking = rank_against_frozen_d(ctx, graded, early_d_sites, d_late)
        calibration = calibrate_against_frozen_d(ctx, graded, ranking["selector"]["ranked"], d_late, final_r, strengths)
        candidates = classify_handles(calibration["calibrated_handles"], graded, args.graded_threshold)
        d_early = choose_early_d(candidates)
        d_refinement = {"candidate_ids": [site.site_id for site in early_d_sites], **ranking, **calibration, "classified_handles": [public_handle(row) for row in candidates]}
    d_chain = [{"name": "D", **public_handle(d_late)}] + ([{"name": "D_early", **public_handle(d_early)}] if d_early else [])
    label_chain(d_chain, "D")

    # Phase 5: handles are frozen. Only now load and use Dte.
    print("[5] Final Dte certification", flush=True)
    load_dte(ctx, coarse)
    load_dte(ctx, graded)
    r_to_y = test_r_to_y(ctx, coarse, r_late)
    r_early_to_late = test_r_edge(ctx, coarse, r_early, r_late) if r_early else None
    d_late_to_r = test_d_edge(ctx, graded, d_late, d_late, final_r, final_r["weights"])
    d_early_to_late = test_d_edge(ctx, graded, d_early, d_late, final_r, d_late["weights"]) if d_early else None
    certification = {"R_to_Y": r_to_y, "R_early_to_R_late": r_early_to_late, "D_late_to_R_to_Y": d_late_to_r, "D_early_to_D_late": d_early_to_late}
    passed_dcal = bool(direct_d["splits"]["Dcal"]["accepted"])
    passed_dte = all(edge is None or edge["accepted"] for edge in certification.values())
    passed = passed_dcal and passed_dte

    coarse_result["heldout"] = [r_to_y]
    if r_refinement is not None:
        r_refinement["heldout"] = [] if r_early_to_late is None else [r_early_to_late]
    if d_refinement is not None:
        d_refinement["heldout"] = [] if d_early_to_late is None else [d_early_to_late]
    direct_d["splits"]["Dte"] = d_late_to_r
    _, direct_d["metrics"] = fit_d(graded, d_late["weights"], SPLITS)
    _, direct_d["R_accuracy"] = fit_r(graded, final_r["weights"], SPLITS)

    final_model = "X -> " + " -> ".join([row["name"] for row in reversed(d_chain)] + [row["name"] for row in reversed(r_chain)]) + " -> Y"
    detailed = {"experiment": "automatic_gradual_PLOT_bracket_v9", "model_info": model_info, "sparse_conversion": [row.to_json() for row in sparse_records], "rules": {"selection": "Dfit/Dcal only", "heldout": "Dte only after all handles are frozen", "support": f"top {args.top_n}, mass >= {args.mass_fraction:g} * top-1"}, "banks": {"coarse": bank_manifest(coarse.examples, coarse.pairs), "graded": bank_manifest(graded.examples, graded.pairs)}, "clean_accuracy": {"coarse": {split: clean_accuracy(coarse.examples, coarse.runs, split=split) for split in SPLITS}, "graded": {split: clean_accuracy(graded.examples, graded.runs, split=split) for split in SPLITS}}, "coarse": coarse_result, "R_refinement": r_refinement, "R_chain": r_chain, "D_discovery": d_discovery, "direct_D_to_R_to_Y": direct_d, "D_refinement": d_refinement, "D_chain": d_chain, "final_model": final_model, "final_certification": certification, "passed_Dcal": passed_dcal, "passed_Dte": passed_dte, "passed": passed}
    summary = {"final_model": final_model, "passed": passed, "passed_Dcal": passed_dcal, "passed_Dte": passed_dte, "selected_handles": {row["name"]: {"sites": row["weights"], "strength": row["strength"]} for row in d_chain + r_chain}, "Dte_edges": {name: short_edge(edge) for name, edge in certification.items()}, "detailed_output": "automatic_gradual_discovery_v9_detailed.json"}

    detailed_path = args.out_dir / "automatic_gradual_discovery_v9_detailed.json"
    summary_path = args.out_dir / "automatic_gradual_discovery_v9_summary.json"
    atomic_json(detailed_path, detailed)
    atomic_json(summary_path, summary)
    if r_refinement is not None:
        atomic_json(args.out_dir / "R_refinement.json", r_refinement)
    atomic_json(args.out_dir / "D_discovery.json", d_discovery)
    if d_refinement is not None:
        atomic_json(args.out_dir / "D_refinement.json", d_refinement)
    print(json.dumps({"status": "complete" if passed else "failed", "final_model": final_model, "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()