from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .ablate_rediscover import (
    HandleConfiguration, # store sites, weights, strength
    abstract_signature, # build abstract effect signature
    atomic_json, # save json
    bank_manifest, # dataset info
    build_bracket_pairs, # bracket intervention pairs
    build_bracket_rediscovery_bank, # bracket dataset
    clean_accuracy, # clean accuracy
    collect_clamped_runs, # collect outputs and site activations
    evaluate_configurations, # intervene on each handle
    load_candidate_circuit, # load 133 candidate sites
    match_signatures, # match abstract and neural signatures with OT
    relation_summary, # metrics for each pair relation
)

from .bracket_progressive_model_discovery import (
    layer_order, # site position in model
)

from .graded_evidence import (
    abstract_e_signature as abstract_d_signature, # abstract signature for depth D
    build_graded_evidence_bank as build_graded_d_bank, # graded depth dataset
    build_graded_pairs, # graded D intervention pairs
    decoder_metrics, # D decoder quality
    e_value as d_value, # get active depth D
    fit_affine_decoder, # fit linear decoder for D
    graded_validation_summary, # D intervention metrics
)

from .progressive_rearly import (
    evaluate_progressive_configurations, # intervene and record downstream sites
    fit_binary_scalar_readout, # fit binary decoder for R
    mediation_summary, # test effect after restoring R
)

from .runtime import (
    load_sparse_gpt_model, # load sparse GPT
    make_tinypython_encoding, # TinyPython tokenizer
)

from .sparse_inference_runtime import (
    convert_transformer_linears_to_sparse, # convert weights to sparse CSR
)


from .print_results import (
    print_discovery_results,
    print_frozen_handle,
    print_refinement,
)

@dataclass
class Bank:
    name: str
    examples: tuple
    pairs: tuple
    by_id: dict
    runs: dict


@dataclass
class Context:
    args: argparse.Namespace
    model: object
    sites: tuple
    site_lookup: dict
    negative_token_id: int
    positive_token_id: int
    device: str


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic gradual discovery of X -> D -> R -> Y.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/automatic_gradual_discovery_v10"))
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-handle-size", type=int, default=2)
    parser.add_argument("--strength-values", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.9)
    parser.add_argument("--invariance-threshold", type=float, default=0.9)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def make_bank(ctx, examples, pairs, name):
    # Build one Dfit, Dcal, or Dte bank and collect its model activations.
    split_examples = tuple(row for row in examples if row.split == name)
    runs = collect_clamped_runs(ctx.model, split_examples, candidate_sites=ctx.sites, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)
    return Bank(name, split_examples, pairs, {row.example_id: row for row in split_examples}, dict(runs))


def run_handles(ctx, bank, handles):
    # Intervene with every handle and return its output margins.
    configs = tuple(HandleConfiguration(row["handle_id"], row["weights"], row["strength"]) for row in handles)
    return evaluate_configurations(ctx.model, configs, bank.pairs, examples=bank.by_id, runs=bank.runs, site_lookup=ctx.site_lookup, disabled_sites=(), hook_means={}, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size)


def run_handles_and_measure(ctx, bank, handles, downstream_sites, restore_sites=()):
    # Intervene with every handle and measure Y plus the frozen downstream sites.
    configs = tuple(HandleConfiguration(row["handle_id"], row["weights"], row["strength"]) for row in handles)
    return evaluate_progressive_configurations(ctx.model, configs, bank.pairs, examples=bank.by_id, runs=bank.runs, site_lookup=ctx.site_lookup, probe_sites=downstream_sites, negative_token_id=ctx.negative_token_id, positive_token_id=ctx.positive_token_id, device=ctx.device, max_batch_size=ctx.args.max_batch_size, restore_probe_site_ids=restore_sites)


def handle_value(run, weights):
    # Compute one handle value from its component sites and weights.
    return sum(float(weight) * float(run.features_by_site[site_id]) for site_id, weight in weights.items())


def handle_order(handle):
    # Use the latest component site as the position of a handle.
    return max(layer_order(site_id) for site_id in handle["weights"])


def get_downstream_sites(ctx, *handles):
    # Get the component sites of the frozen handles that we need to measure.
    site_ids = []
    for handle in handles:
        if handle is None:
            continue
        for site_id in handle["weights"]:
            if site_id not in site_ids:
                site_ids.append(site_id)
    return tuple(ctx.site_lookup[site_id] for site_id in site_ids)


def combine_downstream_values(values, site_ids, weights):
    # Combine measured component-site values into one downstream handle value.
    index = {site_id: i for i, site_id in enumerate(site_ids)}
    return sum(float(weight) * values[..., index[site_id]] for site_id, weight in weights.items())


def normalize_signatures(abstract, neural, pairs):
    # Put abstract and neural signatures on the same scale for OT ranking.
    abstract = torch.tensor(abstract, dtype=torch.float32)
    site_ids = tuple(neural)
    neural_tensor = torch.tensor([neural[site_id] for site_id in site_ids], dtype=torch.float32)
    scales = {}
    for relation in sorted({pair.relation for pair in pairs}):
        indices = [i for i, pair in enumerate(pairs) if pair.relation == relation]
        scale = max(float(torch.sqrt((abstract[indices] ** 2).mean())), float(torch.sqrt((neural_tensor[:, indices] ** 2).mean())), 1e-6)
        abstract[indices] /= scale
        neural_tensor[:, indices] /= scale
        scales[relation] = scale
    neural = {site_id: tuple(float(value) for value in neural_tensor[i]) for i, site_id in enumerate(site_ids)}
    return tuple(float(value) for value in abstract), neural, scales


def rank_sites(ctx, fit_bank, sites, downstream_handle=None):
    # Use Dfit OT mass to rank sites for Y or for one frozen downstream handle.
    single_site_handles = [{"handle_id": site.site_id, "weights": {site.site_id: 1.0}, "strength": 1.0} for site in sites]

    if downstream_handle is None:
        margins = run_handles(ctx, fit_bank, single_site_handles)
        neural = {row["handle_id"]: tuple(float(margins[i, j] - fit_bank.runs[pair.base_id].class_margin) for j, pair in enumerate(fit_bank.pairs)) for i, row in enumerate(single_site_handles)}
        abstract = abstract_signature(fit_bank.pairs, fit_bank.by_id)
        abstract, neural, scales = normalize_signatures(abstract, neural, fit_bank.pairs)
    else:
        downstream_sites = get_downstream_sites(ctx, downstream_handle)
        downstream_site_ids = tuple(site.site_id for site in downstream_sites)
        _, values = run_handles_and_measure(ctx, fit_bank, single_site_handles, downstream_sites)
        scale = downstream_handle["r_readout"].orientation if downstream_handle["variable"] == "R" else downstream_handle["d_decoder"].slope
        neural = {row["handle_id"]: tuple(scale * (float(combine_downstream_values(values[i], downstream_site_ids, downstream_handle["weights"])[j]) - handle_value(fit_bank.runs[pair.base_id], downstream_handle["weights"])) for j, pair in enumerate(fit_bank.pairs)) for i, row in enumerate(single_site_handles)}
        if downstream_handle["variable"] == "R":
            abstract = tuple(float(fit_bank.by_id[pair.source_id].variable_value - fit_bank.by_id[pair.base_id].variable_value) for pair in fit_bank.pairs)
            scales = {}
        else:
            abstract = abstract_d_signature(fit_bank.pairs, fit_bank.by_id, definition="active_depth")
            abstract, neural, scales = normalize_signatures(abstract, neural, fit_bank.pairs)

    selector = match_signatures(abstract, neural, epsilon=ctx.args.selector_epsilon, beta=ctx.args.selector_beta)
    return {"ranked_sites": selector["ranked"], "selector": selector, "normalization_scales": scales}


def get_candidate_pool_sites(ranked_sites, pool_size, mass_fraction):
    # Keep the top OT-ranked sites that also pass the OT-mass cutoff.
    candidate_pool_sites = [dict(row) for row in ranked_sites[:pool_size]]
    if not candidate_pool_sites:
        return []
    cutoff = mass_fraction * float(candidate_pool_sites[0]["weight"])
    return [row for row in candidate_pool_sites if float(row["weight"]) >= cutoff]


def build_candidate_handles(candidate_pool_sites, strength_values, max_handle_size=2):
    # Build every singleton/pair handle from candidate_pool_sites.
    candidate_handles = []
    for k in range(1, max_handle_size + 1):
        for handle_id, sites in enumerate(combinations(candidate_pool_sites, k), start=1):
            total_mass = sum(float(site["weight"]) for site in sites)
            weights = {str(site["site_id"]): float(site["weight"]) / total_mass for site in sites}
            for strength in strength_values:
                candidate_handles.append({"handle_id": f"k{k}_{handle_id}_s{strength:g}", "site_ids": list(weights), "weights": weights, "k": k, "strength": float(strength), "ot_mass": total_mass})
    return candidate_handles


def fit_r(fit_bank, cal_bank, weights):
    # Fit a binary R readout on Dfit and measure R accuracy on Dfit/Dcal.
    readout = fit_binary_scalar_readout([handle_value(fit_bank.runs[row.example_id], weights) for row in fit_bank.examples], [row.variable_value for row in fit_bank.examples])
    accuracy = {}
    for bank in (fit_bank, cal_bank):
        accuracy[bank.name] = float(np.mean([readout.predict(handle_value(bank.runs[row.example_id], weights)) == row.variable_value for row in bank.examples]))
    return readout, accuracy


def fit_d(fit_bank, cal_bank, weights):
    # Fit a graded D decoder on Dfit and measure D Pearson/MAE on Dfit/Dcal.
    decoder = fit_affine_decoder([handle_value(fit_bank.runs[row.example_id], weights) for row in fit_bank.examples], [d_value(row, "active_depth") for row in fit_bank.examples])
    metrics = {}
    for bank in (fit_bank, cal_bank):
        values = [handle_value(bank.runs[row.example_id], weights) for row in bank.examples]
        metrics[bank.name] = decoder_metrics(decoder, values, [d_value(row, "active_depth") for row in bank.examples])
    return decoder, metrics


def add_variable_metrics(handles, fit_bank, cal_bank, graded_threshold):
    # Fit both readouts and label each handle as binary R or graded D.
    results = []
    for handle in handles:
        r_readout, r_accuracy = fit_r(fit_bank, cal_bank, handle["weights"])
        d_decoder, d_metrics = fit_d(fit_bank, cal_bank, handle["weights"])
        is_d = min(abs(float(d_metrics[name]["pearson"])) for name in ("Dfit", "Dcal")) >= graded_threshold
        results.append({**handle, "r_readout": r_readout, "r_accuracy": r_accuracy, "d_decoder": d_decoder, "d_metrics": d_metrics, "is_D": is_d})
    return results


def expected_r(pair, by_id):
    # Return the R value expected after patching source into base.
    base, source = by_id[pair.base_id].variable_value, by_id[pair.source_id].variable_value
    return int(source if source != base else base)


def summarize_r(bank, margins, downstream_values, downstream_handle):
    # Measure sensitivity and invariance for a candidate controlling frozen R.
    by_relation, sensitivity, invariance = defaultdict(list), [], []
    for i, pair in enumerate(bank.pairs):
        base = handle_value(bank.runs[pair.base_id], downstream_handle["weights"])
        source = handle_value(bank.runs[pair.source_id], downstream_handle["weights"])
        patched, target = float(downstream_values[i]), expected_r(pair, bank.by_id)
        row = {"output_correct": (1 if float(margins[i]) > 0 else -1) == target, "downstream_correct": downstream_handle["r_readout"].predict(patched) == target, "downstream_moves": abs(source - patched) < abs(source - base) if abs(source - base) > 1e-8 else abs(patched - base) <= 1e-6}
        by_relation[pair.relation].append(row)
        (sensitivity if bank.by_id[pair.source_id].variable_value != bank.by_id[pair.base_id].variable_value else invariance).append(row)
    relations = {relation: {key: float(np.mean([row[key] for row in rows])) for key in ("output_correct", "downstream_correct", "downstream_moves")} for relation, rows in sorted(by_relation.items())}
    blocks = {"sensitivity_output": float(np.mean([row["output_correct"] for row in sensitivity])), "sensitivity_downstream": float(np.mean([row["downstream_correct"] for row in sensitivity])), "invariance_output": float(np.mean([row["output_correct"] for row in invariance])), "invariance_downstream": float(np.mean([row["downstream_correct"] for row in invariance]))}
    return {"relations": relations, "balanced_blocks": blocks, "score": float(np.mean(list(blocks.values()))), "passes": min(blocks.values()) >= 0.9}


def summarize_d(bank, margins, values, downstream_site_ids, downstream_handle, r_handle):
    # Measure whether a candidate changes frozen D, R, and Y correctly.
    abstract_d = {example_id: d_value(bank.by_id[example_id], "active_depth") for example_id in bank.runs}
    clean_d = {example_id: downstream_handle["d_decoder"].predict(handle_value(run, downstream_handle["weights"])) for example_id, run in bank.runs.items()}
    patched_d = downstream_handle["d_decoder"].slope * combine_downstream_values(values, downstream_site_ids, downstream_handle["weights"]) + downstream_handle["d_decoder"].intercept
    patched_r = [r_handle["r_readout"].predict(value) for value in combine_downstream_values(values, downstream_site_ids, r_handle["weights"])]
    patched_y = np.where(np.asarray(margins) > 0, 1, -1)
    return graded_validation_summary(bank.pairs, bank.by_id, abstract_d, clean_d, patched_d, patched_r, patched_y)


def get_block_scores(summary):
    # Read sensitivity and invariance scores from one evaluation summary.
    blocks = summary.get("balanced_blocks", {})
    sensitivity = [float(value) for key, value in blocks.items() if key.startswith("sensitivity")]
    invariance = [float(value) for key, value in blocks.items() if key.startswith("invariance")]
    return min(sensitivity) if sensitivity else float(summary["score"]), min(invariance) if invariance else float(summary["score"])


def evaluate_handles(ctx, bank, handles, downstream_handle=None, r_handle=None, restore_handle=None):
    # Evaluate handle recovery and edge certification separately.
    if downstream_handle is None:
        margins = run_handles(ctx, bank, handles)
        results = []
        for i, handle in enumerate(handles):
            summary = relation_summary(bank.pairs, bank.by_id, margins[i])
            sensitivity, invariance = get_block_scores(summary)
            recovery_passed = bool(summary["all_rates_at_least_0_90"])
            results.append({**handle, "summary": summary, "sensitivity_score": sensitivity, "invariance_score": invariance, "restoration": None, "recovery_passed": recovery_passed, "restoration_passed": None, "edge_certified": recovery_passed})
        return results

    downstream_sites = get_downstream_sites(ctx, downstream_handle, r_handle if downstream_handle["variable"] == "D" else None)
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    if restore_handle is None:
        restore_handle = downstream_handle
    margins, values = run_handles_and_measure(ctx, bank, handles, downstream_sites)
    restored_margins, _ = run_handles_and_measure(ctx, bank, handles, downstream_sites, tuple(restore_handle["weights"]))
    results = []

    for i, handle in enumerate(handles):
        if downstream_handle["variable"] == "R":
            downstream_values = combine_downstream_values(values[i], downstream_site_ids, downstream_handle["weights"])
            summary = summarize_r(bank, margins[i], downstream_values, downstream_handle)
        else:
            summary = summarize_d(bank, margins[i], values[i], downstream_site_ids, downstream_handle, r_handle)

        restoration = mediation_summary(bank.pairs, bank.by_id, bank.runs, margins[i], restored_margins[i])
        sensitivity, invariance = get_block_scores(summary)
        recovery_passed = bool(summary["passes"] and sensitivity >= ctx.args.sensitivity_threshold and invariance >= ctx.args.invariance_threshold)
        restoration_passed = bool(restoration["passes"])
        results.append({**handle, "summary": summary, "sensitivity_score": sensitivity, "invariance_score": invariance, "restoration": restoration, "recovery_passed": recovery_passed, "restoration_passed": restoration_passed, "edge_certified": recovery_passed and restoration_passed})

    return results


def get_valid_handles(handles, variable, require_restoration=False):
    # Keep handles that recover the requested variable.
    valid = [row for row in handles if row["recovery_passed"]]
    if require_restoration:
        valid = [row for row in valid if row["restoration_passed"]]
    if variable == "R":
        return [row for row in valid if not row["is_D"] and min(row["r_accuracy"].values()) >= 0.9]
    return [row for row in valid if row["is_D"]]


# def removed_fraction_score(row):
#     if row["restoration"] is None:
#         return 0.0
#     return float(row["restoration"]["mean_output_effect_removed_fraction"])
# def handle_selection_key(row):
#     # Rank handles by recovery and restoration scores.
#     return (
#         float(row["summary"]["score"]),
#         float(row["sensitivity_score"]),
#         float(row["invariance_score"]),
#         removed_fraction_score(row),
#         -int(row["k"]),
#         handle_order(row),
#         -abs(float(row["strength"]) - 1.0),
#     )


def restoration_scores(row):
    if row["restoration"] is None:
        return 0.0, 0.0, 0.0

    restoration = row["restoration"]
    return (
        float(restoration["direct_output_matches_source"]),
        float(restoration["restored_Rmid_output_preserves_base"]),
        float(restoration["mean_output_effect_removed_fraction"]),
    )


def handle_selection_key(row):
    # Rank handles by recovery and restoration scores.
    direct_source, restored_base, removed_fraction = restoration_scores(row)
    return (
        float(row["summary"]["score"]),
        float(row["sensitivity_score"]),
        float(row["invariance_score"]),
        direct_source,
        restored_base,
        removed_fraction,
        -int(row["k"]),
        handle_order(row),
        -abs(float(row["strength"]) - 1.0),
    )


def select_late_handle(valid_handles, variable):
    # Select the best passing late handle.
    selected = max(valid_handles, key=handle_selection_key)
    return {**selected, "variable": variable}


def refine_handles(ctx, cal_bank, valid_handles, late_handle, r_handle=None, require_restoration=False):
    # Repeatedly find the strongest earlier handle.
    chain, rounds, current = [late_handle], [], late_handle
    used_sites = set(current["site_ids"])

    while True:
        candidates = [row for row in valid_handles if set(row["site_ids"]).isdisjoint(used_sites) and handle_order(row) < handle_order(current)]
        if not candidates:
            break

        results = evaluate_handles(ctx, cal_bank, candidates, downstream_handle=current, r_handle=r_handle)

        if require_restoration:
            passed = [row for row in results if row["recovery_passed"] and row["restoration_passed"]]
        else:
            passed = [row for row in results if row["recovery_passed"]]

        rounds.append({"downstream_handle": save_handle(current), "results": [save_handle(row) for row in results]})
        if not passed:
            break

        current = max(passed, key=handle_selection_key)
        current = {**current, "variable": late_handle["variable"]}
        chain.append(current)
        used_sites.update(current["site_ids"])

    return chain, rounds


def name_chain(chain, variable):
    # Name the downstream handle late, the upstream handle early, and the rest mid.
    if len(chain) == 1:
        chain[0]["name"] = variable
        return
    chain[0]["name"], chain[-1]["name"] = f"{variable}_late", f"{variable}_early"
    for i in range(1, len(chain) - 1):
        chain[i]["name"] = f"{variable}_mid_{i}"


def certify_chain(ctx, test_bank, chain, r_handle=None):
    # Certify every selected directed edge on Dte without changing the chain.
    variable, results = chain[0]["variable"], []
    if variable == "R":
        result = evaluate_handles(ctx, test_bank, [chain[0]])[0]
        results.append({"edge": f"{chain[0]['name']} -> Y", **result})
    else:
        result = evaluate_handles(ctx, test_bank, [chain[0]], downstream_handle=chain[0], r_handle=r_handle, restore_handle=r_handle)[0]
        results.append({"edge": f"{chain[0]['name']} -> {r_handle['name']} -> Y", **result})
    for i in range(1, len(chain)):
        result = evaluate_handles(ctx, test_bank, [chain[i]], downstream_handle=chain[i - 1], r_handle=r_handle)[0]
        results.append({"edge": f"{chain[i]['name']} -> {chain[i - 1]['name']}", **result})
    return results


def save_handle(handle):
    # Convert readout objects into dictionaries so the handle can be saved as JSON.
    result = {key: value for key, value in handle.items() if key not in ("r_readout", "d_decoder")}
    if "r_readout" in handle:
        result["r_readout"] = handle["r_readout"].to_dict()
    if "d_decoder" in handle:
        result["d_decoder"] = handle["d_decoder"].to_dict()
    return result


def short_handle(handle):
    # Keep only the selected sites, weights, and strength for the summary file.
    return {"sites": handle["site_ids"], "weights": handle["weights"], "strength": handle["strength"]}



def main():
    # Discover R and D on Dfit/Dcal, then load Dte once for final certification.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strength_values = tuple(float(value) for value in args.strength_values.split(","))
    encoder = make_tinypython_encoding(args.circuit_home)
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = convert_transformer_linears_to_sparse(model)
    ctx = Context(args, model, circuit.sites, {site.site_id: site for site in circuit.sites}, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    coarse_examples = build_bracket_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse_fit_bank = make_bank(ctx, coarse_examples, build_bracket_pairs(coarse_examples, split="Dfit", records_per_relation=100), "Dfit")
    coarse_cal_bank = make_bank(ctx, coarse_examples, build_bracket_pairs(coarse_examples, split="Dcal", records_per_relation=100), "Dcal")
    graded_examples = build_graded_d_bank(encoder, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    graded_fit_bank = make_bank(ctx, graded_examples, build_graded_pairs(graded_examples, split="Dfit", records_per_relation=64, e_definition="active_depth"), "Dfit")
    graded_cal_bank = make_bank(ctx, graded_examples, build_graded_pairs(graded_examples, split="Dcal", records_per_relation=48, e_definition="active_depth"), "Dcal")

    # Discover the R chain.
    print("[1] Discover R", flush=True)
    r_ranking = rank_sites(ctx, coarse_fit_bank, ctx.sites)
    r_candidate_pool_sites = get_candidate_pool_sites(r_ranking["ranked_sites"], args.candidate_pool_size, args.mass_fraction)
    r_candidate_handles = build_candidate_handles(r_candidate_pool_sites, strength_values, args.max_handle_size)
    r_candidate_handles = add_variable_metrics(r_candidate_handles, graded_fit_bank, graded_cal_bank, args.graded_threshold)
    r_cal_results = evaluate_handles(ctx, coarse_cal_bank, r_candidate_handles)
    r_valid_handles = get_valid_handles(r_cal_results, "R", require_restoration=False)

    print_discovery_results("R", r_candidate_handles, r_cal_results, r_valid_handles, handle_order, handle_selection_key)
    assert r_valid_handles, "No R handle passed Dcal recovery"

    r_late = select_late_handle(r_valid_handles, "R")
    r_chain, r_refinement = refine_handles(ctx, coarse_cal_bank, r_valid_handles, r_late)
    name_chain(r_chain, "R")
    final_r = r_chain[-1]

    print_refinement("R", r_chain, r_refinement, handle_order)
    print_frozen_handle("R", final_r, handle_order)


    # Discover the D chain.
    print("[2] Discover D", flush=True)
    d_sites = tuple(site for site in ctx.sites if layer_order(site.site_id) < handle_order(final_r) and site.site_id not in final_r["weights"])
    d_ranking = rank_sites(ctx, graded_fit_bank, d_sites, downstream_handle=final_r)
    d_candidate_pool_sites = get_candidate_pool_sites(d_ranking["ranked_sites"], args.candidate_pool_size, args.mass_fraction)
    d_candidate_handles = build_candidate_handles(d_candidate_pool_sites, strength_values, args.max_handle_size)
    d_candidate_handles = add_variable_metrics(d_candidate_handles, graded_fit_bank, graded_cal_bank, args.graded_threshold)
    d_candidate_handles = [{**row, "variable": "D"} for row in d_candidate_handles if row["is_D"]]
    d_cal_results = [evaluate_handles(ctx, graded_cal_bank, [handle], downstream_handle=handle, r_handle=final_r, restore_handle=final_r)[0] for handle in d_candidate_handles]
    d_valid_handles = get_valid_handles(d_cal_results, "D", require_restoration=True)

    print_discovery_results("D", d_candidate_handles, d_cal_results, d_valid_handles, handle_order, handle_selection_key)
    assert d_valid_handles, "No D handle passed Dcal recovery"

    d_late = select_late_handle(d_valid_handles, "D")
    d_chain, d_refinement = refine_handles(ctx, graded_cal_bank, d_valid_handles, d_late, r_handle=final_r)
    name_chain(d_chain, "D")
    final_d = d_chain[-1]

    print_refinement("D", d_chain, d_refinement, handle_order)
    print_frozen_handle("D", final_d, handle_order)

    # Load Dte only after both chains are frozen.
    print("[3] Final Dte certification", flush=True)
    coarse_test_bank = make_bank(ctx, coarse_examples, build_bracket_pairs(coarse_examples, split="Dte", records_per_relation=100), "Dte")
    graded_test_bank = make_bank(ctx, graded_examples, build_graded_pairs(graded_examples, split="Dte", records_per_relation=64, e_definition="active_depth"), "Dte")
    r_test_results = certify_chain(ctx, coarse_test_bank, r_chain)
    d_test_results = certify_chain(ctx, graded_test_bank, d_chain, r_handle=final_r)
    passed = all(row["edge_certified"] for row in r_test_results + d_test_results)

    final_model = "X -> " + " -> ".join([row["name"] for row in reversed(d_chain)] + [row["name"] for row in reversed(r_chain)]) + " -> Y"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    detailed = {
        "experiment": "automatic_gradual_discovery_v10", "config": config, "model_info": model_info, "sparse_conversion": [row.to_json() for row in sparse_records],
        "R": {"ranking": r_ranking, "candidate_pool_sites": r_candidate_pool_sites, "candidate_handles": [save_handle(row) for row in r_candidate_handles], "cal_results": [save_handle(row) for row in r_cal_results], "valid_handles": [save_handle(row) for row in r_valid_handles], "refinement": r_refinement, "chain": [save_handle(row) for row in r_chain]},
        "D": {"ranking": d_ranking, "candidate_pool_sites": d_candidate_pool_sites, "candidate_handles": [save_handle(row) for row in d_candidate_handles], "cal_results": [save_handle(row) for row in d_cal_results], "valid_handles": [save_handle(row) for row in d_valid_handles], "refinement": d_refinement, "chain": [save_handle(row) for row in d_chain]},
        "Dte": {"R_edges": [save_handle(row) for row in r_test_results], "D_edges": [save_handle(row) for row in d_test_results], "passed": passed},
        "banks": {"coarse": bank_manifest(coarse_examples, {"Dfit": coarse_fit_bank.pairs, "Dcal": coarse_cal_bank.pairs, "Dte": coarse_test_bank.pairs}), "graded": bank_manifest(graded_examples, {"Dfit": graded_fit_bank.pairs, "Dcal": graded_cal_bank.pairs, "Dte": graded_test_bank.pairs})},
        "clean_accuracy": {"coarse": {bank.name: clean_accuracy(bank.examples, bank.runs, split=bank.name) for bank in (coarse_fit_bank, coarse_cal_bank, coarse_test_bank)}, "graded": {bank.name: clean_accuracy(bank.examples, bank.runs, split=bank.name) for bank in (graded_fit_bank, graded_cal_bank, graded_test_bank)}},
        "final_model": final_model, "passed": passed,
    }
    summary = {"final_model": final_model, "passed": passed, "handles": {row["name"]: short_handle(row) for row in d_chain + r_chain}, "Dte_edges": {row["edge"]: row["edge_certified"] for row in r_test_results + d_test_results}, "detailed_output": "automatic_gradual_discovery_v10_detailed.json"}
    detailed_path = args.out_dir / "automatic_gradual_discovery_v10_detailed.json"
    summary_path = args.out_dir / "automatic_gradual_discovery_v10_summary.json"
    atomic_json(detailed_path, detailed)
    atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()