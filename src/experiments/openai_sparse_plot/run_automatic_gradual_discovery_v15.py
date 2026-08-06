from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

from . import run_automatic_gradual_discovery_v10 as discovery
from .plot_matching import cost_matrix


EXPERIMENT_NAME = "automatic_gradual_discovery_v15"


def parse_args():
    parser = argparse.ArgumentParser(description="Three-stage causal variable discovery with final graph evaluation.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"outputs/{EXPERIMENT_NAME}"))
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-handle-size", type=int, default=2)
    parser.add_argument("--strength-values", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.9)
    parser.add_argument("--invariance-threshold", type=float, default=0.9)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def save_handles(handles):
    return [discovery.save_handle(handle) for handle in handles]


# Step 1: candidate site selection
def match_signatures_cosine(abstract, neural):
    site_ids = tuple(neural)
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural[site_id]) for site_id in site_ids], dtype=torch.float32)

    # Cosine similarity is the site-selection score.
    costs = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    similarities = 1.0 - costs[0]
    weights = similarities.clamp_min(0.0)

    if float(weights.sum()) <= 0.0:
        weights = torch.softmax(similarities, dim=0)
    else:
        weights = weights / weights.sum()

    ranked = []
    for index, site_id in enumerate(site_ids):
        ranked.append({"site_id": site_id, "weight": float(weights[index]), "similarity": float(similarities[index]), "cost": float(costs[0, index])})

    ranked.sort(key=lambda row: (-row["similarity"], row["site_id"]))
    return {"cost_mode": "raw_cosine", "matching": "direct_cosine_similarity", "ranked": ranked}


def rank_sites(ctx, fit_bank, sites, downstream_handle=None):
    # Intervene on every site independently to get its neural signature.
    single_site_handles = [{"handle_id": site.site_id, "weights": {site.site_id: 1.0}, "strength": 1.0} for site in sites]
    neural_signatures = {}

    if downstream_handle is None:
        # No frozen downstream handle: measure each site's effect at the output.
        output_margins = discovery.run_handles(ctx, fit_bank, single_site_handles)

        for handle_index, handle in enumerate(single_site_handles):
            site_signature = []
            for pair_index, pair in enumerate(fit_bank.pairs):
                patched_margin = float(output_margins[handle_index, pair_index])
                base_margin = float(fit_bank.runs[pair.base_id].class_margin)
                site_signature.append(patched_margin - base_margin)
            neural_signatures[handle["handle_id"]] = tuple(site_signature)

        abstract_signature = discovery.abstract_signature(fit_bank.pairs, fit_bank.by_id)
        abstract_signature, neural_signatures, normalization_scales = discovery.normalize_signatures(abstract_signature, neural_signatures, fit_bank.pairs)

    else:
        # Frozen downstream handle: measure each site's effect there instead of at Y.
        downstream_sites = discovery.get_downstream_sites(ctx, downstream_handle)
        downstream_site_ids = tuple(site.site_id for site in downstream_sites)
        _, measured_downstream_values = discovery.run_handles_and_measure(ctx, fit_bank, single_site_handles, downstream_sites)
        decoder_scale = downstream_handle["r_readout"].orientation if downstream_handle["variable"] == "R" else downstream_handle["d_decoder"].slope

        for handle_index, handle in enumerate(single_site_handles):
            combined_values = discovery.combine_downstream_values(measured_downstream_values[handle_index], downstream_site_ids, downstream_handle["weights"])
            site_signature = []
            for pair_index, pair in enumerate(fit_bank.pairs):
                patched_value = float(combined_values[pair_index])
                base_run = fit_bank.runs[pair.base_id]
                base_value = discovery.handle_value(base_run, downstream_handle["weights"])
                site_signature.append(decoder_scale * (patched_value - base_value))
            neural_signatures[handle["handle_id"]] = tuple(site_signature)

        # Build the abstract signature of the causal variable on the same pairs.
        if downstream_handle["variable"] == "R":
            abstract_values = []
            for pair in fit_bank.pairs:
                base_example = fit_bank.by_id[pair.base_id]
                source_example = fit_bank.by_id[pair.source_id]
                abstract_values.append(float(source_example.variable_value) - float(base_example.variable_value))
            abstract_signature = tuple(abstract_values)
            normalization_scales = {}
        else:
            abstract_signature = discovery.abstract_d_signature(fit_bank.pairs, fit_bank.by_id, definition="active_depth")
            abstract_signature, neural_signatures, normalization_scales = discovery.normalize_signatures(abstract_signature, neural_signatures, fit_bank.pairs)

    # Rank sites by alignment between their neural signature and the abstract one.
    selector_result = match_signatures_cosine(abstract_signature, neural_signatures)
    return {"ranked_sites": selector_result["ranked"], "selector": selector_result, "normalization_scales": normalization_scales}


def select_candidate_sites(ctx, fit_bank, sites, downstream_handle=None):
    # Rank sites by cosine similarity, then keep only the top sites.
    ranking = rank_sites(ctx, fit_bank, sites, downstream_handle=downstream_handle)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)
    return ranking, pool


# Step 2: candidate handle construction
def build_handles(ctx, supports, pool, strengths, variable, fit_bank, cal_bank):
    # Build one weighted handle for each support.
    site_rows = {str(row["site_id"]): row for row in pool}

    base_handles = []
    for index, support in enumerate(supports, 1):
        total_mass = sum(float(site_rows[site_id]["weight"]) for site_id in support)

        weights = {}
        for site_id in support:
            weights[site_id] = float(site_rows[site_id]["weight"]) / total_mass if total_mass > 0.0 else 1.0 / len(support)

        base_handles.append({"handle_id": f"k{len(support)}_{index}", "site_ids": list(support), "weights": weights, "k": len(support), "strength": 1.0, "ot_mass": total_mass})

    # Fit the variable decoder/readout once, then test all intervention strengths.
    fitted = discovery.add_variable_metrics(base_handles, fit_bank, cal_bank, ctx.args.graded_threshold)
    handles = []
    for row in fitted:
        for strength in strengths:
            handle = dict(row)
            handle["handle_id"] = f"{row['handle_id']}_s{strength:g}"
            handle["strength"] = float(strength)
            handle["variable"] = variable
            handles.append(handle)
    return handles


def evaluate_handles(ctx, bank, handles, downstream=None, r_handle=None):
    return discovery.evaluate_handles(ctx, bank, handles, downstream_handle=downstream, r_handle=r_handle)


def construct_candidate_handles(ctx, pool, strengths, variable, fit_bank, metric_cal_bank, intervention_bank, downstream=None, r_handle=None):
    # Start from singleton supports.
    supports = [(str(row["site_id"]),) for row in pool]
    site_order = {str(row["site_id"]): index for index, row in enumerate(pool)}

    all_handles = []
    all_results = []
    all_valid = []
    levels = []

    for size in range(1, ctx.args.max_handle_size + 1):
        if not supports:
            break

        handles = build_handles(ctx, supports, pool, strengths, variable, fit_bank, metric_cal_bank)
        results = evaluate_handles(ctx, intervention_bank, handles, downstream=downstream, r_handle=r_handle)

        # Keep configurations that pass recovery and represent the current variable.
        valid_configurations = discovery.get_valid_handles(results, variable, require_restoration=False)

        # Several strengths can pass for one support. Keep only the best strength.
        valid_by_support = {}
        for row in valid_configurations:
            valid_by_support.setdefault(tuple(row["site_ids"]), []).append(row)
        best_by_support = {support: max(rows, key=discovery.handle_selection_key) for support, rows in valid_by_support.items()}
        valid_handles = list(best_by_support.values())
        failed = [support for support in supports if support not in valid_by_support]

        # Sites inside any valid handle cannot be used to build larger handles.
        removed_sites = {site_id for handle in valid_handles for site_id in handle["site_ids"]}

        all_handles.extend(handles)
        all_results.extend(results)
        all_valid.extend(valid_handles)

        level = {
            "k": size,
            "supports": [list(support) for support in supports],
            "valid_handles": save_handles(valid_handles),
            "failed_supports": [list(support) for support in failed],
            "removed_sites": sorted(removed_sites),
        }
        levels.append(level)

        print(f"  k={size}: supports={len(supports)}, valid={len(valid_handles)}, failed={len(failed)}", flush=True)

        # Only failed supports can be combined into the next size.
        failed_for_next = [support for support in failed if not removed_sites.intersection(support)]
        failed_set = set(failed_for_next)
        next_size = size + 1
        joined = set()

        for left, right in combinations(failed_for_next, 2):
            site_ids = set(left) | set(right)
            if len(site_ids) != next_size:
                continue

            support = tuple(sorted(site_ids, key=lambda site_id: site_order[site_id]))
            if all(subset in failed_set for subset in combinations(support, size)):
                joined.add(support)

        supports = sorted(joined, key=lambda support: tuple(site_order[site_id] for site_id in support))

    return all_handles, all_results, all_valid, levels


def choose_strongest_handle(handles):
    # Scores decide first. Position is used only when the full score key ties.
    best_score = max(discovery.handle_selection_key(handle) for handle in handles)
    tied = [handle for handle in handles if discovery.handle_selection_key(handle) == best_score]
    return max(tied, key=discovery.handle_order)


# Step 3: causal graph construction
def name_nodes(handles, variable):
    nodes = [dict(handle) for handle in handles]
    nodes.sort(key=discovery.handle_order)

    if len(nodes) == 1:
        nodes[0]["name"] = variable
        return nodes

    nodes[0]["name"] = f"{variable}_early"
    nodes[-1]["name"] = f"{variable}_late"
    for index, node in enumerate(reversed(nodes[1:-1]), 1):
        node["name"] = f"{variable}_mid_{index}"
    return nodes


def downstream_passes(ctx, result):
    sensitivity_passed = float(result["sensitivity_score"]) >= ctx.args.sensitivity_threshold
    invariance_passed = float(result["invariance_score"]) >= ctx.args.invariance_threshold
    return bool(result["recovery_passed"] and sensitivity_passed and invariance_passed)


def make_graph_edge(source, downstream_name, downstream_handle, result, cached_direct, evaluation_mode):
    # Restoration does not decide the main edge. A failure only adds a possible bypass to Y.
    bypass_to_y = downstream_name != "Y" and not bool(result.get("restoration_passed"))
    return {
        "source": source["name"],
        "downstream": downstream_name,
        "source_handle": source,
        "downstream_handle": downstream_handle,
        "discovery_result": result,
        "cached_direct": cached_direct,
        "evaluation_mode": evaluation_mode,
        "bypass_to_y": bypass_to_y,
    }


def build_chain_edges(nodes, terminal_name, terminal_handle=None):
    # Each chosen handle was already validated against the frozen handle from the previous round.
    edges = []
    for index, source in enumerate(nodes):
        if index + 1 < len(nodes):
            downstream = nodes[index + 1]
            edges.append(make_graph_edge(source, downstream["name"], downstream, source, True, "downstream"))
        elif terminal_handle is not None:
            edges.append(make_graph_edge(source, terminal_name, terminal_handle, source, True, "downstream"))
        else:
            edges.append(make_graph_edge(source, terminal_name, None, source, True, "output"))
    return edges


def expanded_edge_labels(edges):
    labels = []
    for edge in edges:
        labels.append(f"{edge['source']} -> {edge['downstream']}")
        if edge["bypass_to_y"]:
            labels.append(f"{edge['source']} -> Y")
    return labels


def save_graph_edges(edges):
    saved = []
    for edge in edges:
        row = {
            "edge": f"{edge['source']} -> {edge['downstream']}",
            "source": edge["source"],
            "downstream": edge["downstream"],
            "cached_direct": edge["cached_direct"],
            "evaluation_mode": edge["evaluation_mode"],
            "downstream_recovery_passed": True,
            "restoration_passed": bool(edge["discovery_result"].get("restoration_passed")),
            "bypass_to_Y": edge["bypass_to_y"],
            "source_handle": discovery.save_handle(edge["source_handle"]),
            "discovery_result": discovery.save_handle(edge["discovery_result"]),
        }
        if edge["downstream_handle"] is not None:
            row["downstream_handle"] = discovery.save_handle(edge["downstream_handle"])
        saved.append(row)
    return saved


# Step 5: final graph evaluation
def evaluate_graph(ctx, bank, edges, r_handle=None):
    # Step 5: rerun every discovered edge on held-out test data.
    results = []
    for edge in edges:
        source = edge["source_handle"]
        if edge["evaluation_mode"] == "output":
            result = evaluate_handles(ctx, bank, [source])[0]
        else:
            result = evaluate_handles(ctx, bank, [source], downstream=edge["downstream_handle"], r_handle=r_handle)[0]

        heldout_recovery = downstream_passes(ctx, result)
        heldout_bypass = edge["downstream"] != "Y" and not bool(result.get("restoration_passed"))
        structure_passed = heldout_recovery and heldout_bypass == edge["bypass_to_y"]

        row = dict(result)
        row["edge"] = f"{edge['source']} -> {edge['downstream']}"
        row["downstream_recovery_passed"] = heldout_recovery
        row["discovered_bypass_to_Y"] = edge["bypass_to_y"]
        row["heldout_bypass_to_Y"] = heldout_bypass
        row["graph_structure_passed"] = structure_passed
        results.append(row)
    return results


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = tuple(float(value) for value in args.strength_values.split(","))

    # Load model and candidate circuit.
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)
    site_lookup = {site.site_id: site for site in circuit.sites}
    ctx = discovery.Context(args, model, circuit.sites, site_lookup, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    # Build discovery/calibration banks. Dte is created only at final evaluation.
    coarse_examples = discovery.build_bracket_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse_fit_pairs = discovery.build_bracket_pairs(coarse_examples, split="Dfit", records_per_relation=100)
    coarse_cal_pairs = discovery.build_bracket_pairs(coarse_examples, split="Dcal", records_per_relation=100)
    coarse_fit = discovery.make_bank(ctx, coarse_examples, coarse_fit_pairs, "Dfit")
    coarse_cal = discovery.make_bank(ctx, coarse_examples, coarse_cal_pairs, "Dcal")

    graded_examples = discovery.build_graded_d_bank(encoder, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    graded_fit_pairs = discovery.build_graded_pairs(graded_examples, split="Dfit", records_per_relation=64, e_definition="active_depth")
    graded_cal_pairs = discovery.build_graded_pairs(graded_examples, split="Dcal", records_per_relation=48, e_definition="active_depth")
    graded_fit = discovery.make_bank(ctx, graded_examples, graded_fit_pairs, "Dfit")
    graded_cal = discovery.make_bank(ctx, graded_examples, graded_cal_pairs, "Dcal")

    # Discover R progressively: Y <- R_late <- R_mid <- ... <- R_early.
    r_rounds = []
    r_selected = []
    r_frozen_sites = set()
    r_downstream = None
    round_index = 1

    while True:
        if r_downstream is None:
            r_sites = [site for site in ctx.sites if site.site_id not in r_frozen_sites]
        else:
            downstream_order = discovery.handle_order(r_downstream)
            r_sites = [site for site in ctx.sites if discovery.layer_order(site.site_id) < downstream_order and site.site_id not in r_frozen_sites]
        if not r_sites:
            break

        print(f"[R round {round_index}/1] Candidate site selection", flush=True)
        r_ranking, r_pool = select_candidate_sites(ctx, coarse_fit, tuple(r_sites), downstream_handle=r_downstream)

        print(f"[R round {round_index}/2] Candidate handle construction", flush=True)
        r_handles, r_results, r_valid, r_levels = construct_candidate_handles(ctx, r_pool, strengths, "R", graded_fit, graded_cal, coarse_cal, downstream=r_downstream)

        round_result = {
            "round": round_index,
            "frozen_downstream_handle": None if r_downstream is None else discovery.save_handle(r_downstream),
            "candidate_site_selection": {"ranking": r_ranking, "candidate_pool_sites": r_pool},
            "candidate_handle_construction": {"search_levels": r_levels, "candidate_handles": save_handles(r_handles), "validation_results": save_handles(r_results), "valid_handles": save_handles(r_valid)},
            "chosen_handle": None,
        }

        if not r_valid:
            r_rounds.append(round_result)
            break

        chosen = choose_strongest_handle(r_valid)
        round_result["chosen_handle"] = discovery.save_handle(chosen)
        r_rounds.append(round_result)
        r_selected.append(chosen)
        r_frozen_sites.update(chosen["site_ids"])
        r_downstream = chosen
        round_index += 1

    if not r_selected:
        raise RuntimeError("No valid R handle")

    print("[R/3] Causal graph construction", flush=True)
    r_nodes = name_nodes(r_selected, "R")
    r_edges = build_chain_edges(r_nodes, "Y")
    r_early = r_nodes[0]

    # Discover D progressively upstream of R_early.
    d_rounds = []
    d_selected = []
    d_frozen_sites = {site_id for node in r_nodes for site_id in node["site_ids"]}
    d_downstream = r_early
    round_index = 1

    while True:
        downstream_order = discovery.handle_order(d_downstream)
        d_sites = [site for site in ctx.sites if discovery.layer_order(site.site_id) < downstream_order and site.site_id not in d_frozen_sites]
        if not d_sites:
            break

        print(f"[D round {round_index}/1] Candidate site selection", flush=True)
        d_ranking, d_pool = select_candidate_sites(ctx, graded_fit, tuple(d_sites), downstream_handle=d_downstream)

        print(f"[D round {round_index}/2] Candidate handle construction", flush=True)
        d_handles, d_results, d_valid, d_levels = construct_candidate_handles(ctx, d_pool, strengths, "D", graded_fit, graded_cal, graded_cal, downstream=d_downstream, r_handle=r_early)

        round_result = {
            "round": round_index,
            "frozen_downstream_handle": discovery.save_handle(d_downstream),
            "candidate_site_selection": {"ranking": d_ranking, "candidate_pool_sites": d_pool},
            "candidate_handle_construction": {"search_levels": d_levels, "candidate_handles": save_handles(d_handles), "validation_results": save_handles(d_results), "valid_handles": save_handles(d_valid)},
            "chosen_handle": None,
        }

        if not d_valid:
            d_rounds.append(round_result)
            break

        chosen = choose_strongest_handle(d_valid)
        round_result["chosen_handle"] = discovery.save_handle(chosen)
        d_rounds.append(round_result)
        d_selected.append(chosen)
        d_frozen_sites.update(chosen["site_ids"])
        d_downstream = chosen
        round_index += 1

    if not d_selected:
        raise RuntimeError("No valid D handle")

    print("[D/3] Causal graph construction", flush=True)
    d_nodes = name_nodes(d_selected, "D")
    d_edges = build_chain_edges(d_nodes, r_early["name"], terminal_handle=r_early)

    # Step 4 is intentionally not used in v14.
    print("[4] Graph pruning skipped (future improvement)", flush=True)

    # Step 5: evaluate the complete discovered graph only once on Dte.
    print("[5] Final graph evaluation on Dte", flush=True)
    coarse_test_pairs = discovery.build_bracket_pairs(coarse_examples, split="Dte", records_per_relation=100)
    graded_test_pairs = discovery.build_graded_pairs(graded_examples, split="Dte", records_per_relation=64, e_definition="active_depth")
    coarse_test = discovery.make_bank(ctx, coarse_examples, coarse_test_pairs, "Dte")
    graded_test = discovery.make_bank(ctx, graded_examples, graded_test_pairs, "Dte")
    r_test_results = evaluate_graph(ctx, coarse_test, r_edges)
    d_test_results = evaluate_graph(ctx, graded_test, d_edges, r_handle=r_early)
    passed = all(row["graph_structure_passed"] for row in r_test_results + d_test_results)

    graph_edges = expanded_edge_labels(d_edges)
    graph_edges.extend(expanded_edge_labels(r_edges))
    final_model = "; ".join(graph_edges)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    sparse_conversion = [row.to_json() for row in sparse_records]

    detailed = {
        "experiment": EXPERIMENT_NAME,
        "config": config,
        "model_info": model_info,
        "sparse_conversion": sparse_conversion,
        "R": {
            "progressive_rounds": r_rounds,
            "causal_graph_construction": {"nodes": save_handles(r_nodes), "edges": save_graph_edges(r_edges), "expanded_edges": expanded_edge_labels(r_edges)},
        },
        "D": {
            "progressive_rounds": d_rounds,
            "causal_graph_construction": {"nodes": save_handles(d_nodes), "edges": save_graph_edges(d_edges), "expanded_edges": expanded_edge_labels(d_edges)},
        },
        "graph_pruning": {"enabled": False, "status": "future_improvement"},
        "final_graph_evaluation": {"split": "Dte", "R_edges": save_handles(r_test_results), "D_edges": save_handles(d_test_results), "passed": passed},
        "final_model": final_model,
        "passed": passed,
    }

    handles = {node["name"]: discovery.short_handle(node) for node in d_nodes + r_nodes}
    detailed_name = f"{EXPERIMENT_NAME}_detailed.json"
    summary_name = f"{EXPERIMENT_NAME}_summary.json"
    summary = {"final_model": final_model, "passed": passed, "handles": handles, "R_edges": save_graph_edges(r_edges), "D_edges": save_graph_edges(d_edges), "detailed_output": detailed_name}
    detailed_path = args.out_dir / detailed_name
    summary_path = args.out_dir / summary_name
    discovery.atomic_json(detailed_path, detailed)
    discovery.atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()