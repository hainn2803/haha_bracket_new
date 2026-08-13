from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch

from . import run_automatic_gradual_discovery_v10 as discovery
from .plot_matching import cost_matrix


EXPERIMENT_NAME = "automatic_gradual_discovery_v18"
COSINE_TOLERANCE = 0.001


def parse_args():
    # Configure progressive discovery and held-out evaluation for the bracket task.
    parser = argparse.ArgumentParser(description="Progressive causal discovery for the closing-bracket task.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"outputs/{EXPERIMENT_NAME}"))
    parser.add_argument("--candidate-pool-size", type=int, default=4)
    parser.add_argument("--max-handle-size", type=int, default=2)
    parser.add_argument("--strength-values", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--signature-threshold", type=float, default=0.9)
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.9)
    parser.add_argument("--invariance-threshold", type=float, default=0.9)
    parser.add_argument("--restoration-direct-threshold", type=float, default=0.9)
    parser.add_argument("--restoration-base-threshold", type=float, default=0.9)
    parser.add_argument("--restoration-removed-threshold", type=float, default=0.5)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def save_handles(handles):
    # Convert handles to JSON-safe dictionaries.
    return [discovery.save_handle(handle) for handle in handles]


# Signature matching
def match_signatures_cosine(abstract, neural):
    # Compare every neural signature with the abstract signature and convert cosine to mass.
    candidate_ids = tuple(neural)
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural[candidate_id]) for candidate_id in candidate_ids], dtype=torch.float32)

    # Cosine similarity is the site-selection score.
    costs = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    similarities = 1.0 - costs[0]
    weights = similarities.clamp_min(0.0)

    if float(weights.sum()) <= 0.0:
        weights = torch.softmax(similarities, dim=0)
    else:
        weights = weights / weights.sum()

    ranked = []
    for index, candidate_id in enumerate(candidate_ids):
        ranked.append({"candidate_id": candidate_id, "weight": float(weights[index]), "similarity": float(similarities[index]), "cost": float(costs[0, index])})

    ranked.sort(key=lambda row: (-row["similarity"], row["candidate_id"]))
    return {"cost_mode": "raw_cosine", "matching": "direct_cosine_similarity", "ranked": ranked}


def build_chain_signatures(ctx, bank, handles, frozen_chain):
    # Concatenate each candidate's effects on every frozen handle and on Y.
    neural = {handle["handle_id"]: [] for handle in handles}
    abstract = []

    # Collect all component sites in the frozen chain.
    downstream_sites = []
    seen_site_ids = set()
    for frozen in frozen_chain:
        for site in discovery.get_downstream_sites(ctx, frozen):
            if site.site_id not in seen_site_ids:
                downstream_sites.append(site)
                seen_site_ids.add(site.site_id)

    # Patch every candidate once and measure the complete frozen chain and Y.
    if downstream_sites:
        output_margins, measured_values = discovery.run_handles_and_measure(ctx, bank, handles, tuple(downstream_sites))
    else:
        output_margins = discovery.run_handles(ctx, bank, handles)
        measured_values = None

    measured_site_ids = tuple(site.site_id for site in downstream_sites)

    def append_block(abstract_block, neural_block, reference_block):
        # Normalize the abstract block and scale every candidate by the same clean source-base reference.
        abstract_tensor = torch.tensor(abstract_block, dtype=torch.float32)
        reference_tensor = torch.tensor(reference_block, dtype=torch.float32)
        active = abstract_tensor.abs() > 0.0
        abstract_norm = float(torch.linalg.vector_norm(abstract_tensor))
        reference_norm = float(torch.linalg.vector_norm(reference_tensor[active]))

        if abstract_norm <= 0.0 or reference_norm <= 0.0:
            raise ValueError("Abstract and reference signature blocks must have nonzero norm")

        abstract.extend(float(value) for value in abstract_tensor / abstract_norm)

        for candidate_id, values in neural_block.items():
            values_tensor = torch.tensor(values, dtype=torch.float32)
            neural[candidate_id].extend(float(value) for value in values_tensor / reference_norm)

    # Add one block for each frozen R or D handle, from the nearest downstream handle toward Y.
    for frozen in frozen_chain:
        frozen_site_ids = tuple(frozen["weights"])
        indices = [measured_site_ids.index(site_id) for site_id in frozen_site_ids]
        decoder_scale = frozen["r_readout"].orientation if frozen["variable"] == "R" else frozen["d_decoder"].slope
        reference_block = []
        neural_block = {}

        for pair in bank.pairs:
            source_value = discovery.handle_value(bank.runs[pair.source_id], frozen["weights"])
            base_value = discovery.handle_value(bank.runs[pair.base_id], frozen["weights"])
            reference_block.append(decoder_scale * (float(source_value) - float(base_value)))

        for handle_index, handle in enumerate(handles):
            values = measured_values[handle_index][..., indices]
            combined_values = discovery.combine_downstream_values(values, frozen_site_ids, frozen["weights"])
            signature = []
            for pair_index, pair in enumerate(bank.pairs):
                base_value = discovery.handle_value(bank.runs[pair.base_id], frozen["weights"])
                signature.append(decoder_scale * (float(combined_values[pair_index]) - float(base_value)))
            neural_block[handle["handle_id"]] = tuple(signature)

        if frozen["variable"] == "R":
            abstract_block = tuple(float(bank.by_id[pair.source_id].variable_value) - float(bank.by_id[pair.base_id].variable_value) for pair in bank.pairs)
        else:
            abstract_block = discovery.abstract_d_signature(bank.pairs, bank.by_id, definition="active_depth")
        append_block(abstract_block, neural_block, reference_block)

    # Add the output-margin effect as the final block.
    output_reference_block = []
    output_block = {}

    for pair in bank.pairs:
        source_margin = float(bank.runs[pair.source_id].class_margin)
        base_margin = float(bank.runs[pair.base_id].class_margin)
        output_reference_block.append(source_margin - base_margin)

    for handle_index, handle in enumerate(handles):
        output_signature = []
        for pair_index, pair in enumerate(bank.pairs):
            patched_margin = float(output_margins[handle_index, pair_index])
            base_margin = float(bank.runs[pair.base_id].class_margin)
            output_signature.append(patched_margin - base_margin)
        output_block[handle["handle_id"]] = tuple(output_signature)
    append_block(discovery.abstract_signature(bank.pairs, bank.by_id), output_block, output_reference_block)

    return tuple(abstract), {candidate_id: tuple(values) for candidate_id, values in neural.items()}


def rank_handles_by_signature(ctx, bank, handles, frozen_chain):
    # Give each handle configuration one cosine score and normalized mass.
    abstract_signature, neural_signatures = build_chain_signatures(ctx, bank, handles, frozen_chain)
    selector = match_signatures_cosine(abstract_signature, neural_signatures)
    scores = {row["candidate_id"]: row for row in selector["ranked"]}

    ranked = []
    for handle in handles:
        row = dict(handle)
        score = scores[handle["handle_id"]]
        row["signature_similarity"] = score["similarity"]
        row["signature_mass"] = score["weight"]
        row["signature_cost"] = score["cost"]
        ranked.append(row)

    ranked.sort(key=lambda row: (row["signature_mass"], discovery.handle_order(row), -row["k"], -abs(row["strength"] - 1.0)), reverse=True)
    return ranked, selector


# Step 1: candidate site selection
def rank_sites(ctx, bank, sites, frozen_chain):
    # Screen eligible sites using singleton signatures at strength 1.
    singletons = [{"handle_id": site.site_id, "site_ids": [site.site_id], "weights": {site.site_id: 1.0}, "k": 1, "strength": 1.0} for site in sites]
    ranked_handles, selector = rank_handles_by_signature(ctx, bank, singletons, frozen_chain)

    ranked_sites = []
    for handle in ranked_handles:
        ranked_sites.append({"site_id": handle["site_ids"][0], "weight": handle["signature_mass"], "similarity": handle["signature_similarity"], "cost": handle["signature_cost"]})
    return {"ranked_sites": ranked_sites, "selector": selector, "chain_depth": len(frozen_chain)}


def select_candidate_sites(ctx, bank, sites, variable, graded_fit, graded_cal, frozen_chain):
    # Rank all sites by cosine and take Top-K before classifying them as R or D.
    ranking = rank_sites(ctx, bank, sites, frozen_chain)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)

    singletons = [{"handle_id": str(row["site_id"]), "site_ids": [str(row["site_id"])], "weights": {str(row["site_id"]): 1.0}, "k": 1, "strength": 1.0} for row in pool]

    classified = discovery.add_variable_metrics(singletons, graded_fit, graded_cal, ctx.args.graded_threshold)
    is_d = {row["site_ids"][0]: bool(row["is_D"]) for row in classified}

    for row in pool:
        row["is_D"] = is_d[str(row["site_id"])]
        row["selected_for_variable"] = not row["is_D"] if variable == "R" else row["is_D"]

    print(f"  Top-{len(pool)} classified sites for {variable}:", flush=True)
    for row in pool:
        print(f"    site={row['site_id']}, cosine={row['similarity']:.4f}, mass={row['weight']:.4f}, is_D={row['is_D']}, selected={row['selected_for_variable']}", flush=True)

    ranking["top_k_classified"] = pool
    return ranking, [row for row in pool if row["selected_for_variable"]]


# Step 2: candidate handle construction and signature ranking
def build_handles(ctx, pool, strengths, variable):
    # Build singleton/pair/... supports and instantiate every requested strength.
    site_rows = {str(row["site_id"]): row for row in pool}
    handles = []
    support_index = 1

    for size in range(1, ctx.args.max_handle_size + 1):
        for support in combinations(site_rows, size):
            total_mass = sum(float(site_rows[site_id]["weight"]) for site_id in support)
            weights = {}
            for site_id in support:
                if total_mass > 0.0:
                    weights[site_id] = float(site_rows[site_id]["weight"]) / total_mass
                else:
                    weights[site_id] = 1.0 / size

            for strength in strengths:
                handles.append({
                    "handle_id": f"k{size}_{support_index}_s{strength:g}",
                    "site_ids": list(support),
                    "weights": weights,
                    "k": size,
                    "strength": float(strength),
                    "variable": variable,
                    "site_mass": total_mass,
                })
            support_index += 1
    return handles


def evaluate_handles(ctx, bank, handles, downstream=None, r_handle=None):
    # Run bracket recovery and restoration for R or D handles.
    return discovery.evaluate_handles(ctx, bank, handles, downstream_handle=downstream, r_handle=r_handle)


def add_discovery_diagnostics(ctx, fit_bank, cal_bank, handles, downstream, r_handle=None):
    # Attach Dcal recovery and Dfit/Dcal restoration diagnostics to every handle.
    if not handles:
        return handles

    cal_results = evaluate_handles(ctx, cal_bank, handles, downstream=downstream, r_handle=r_handle)
    fit_results = None if downstream is None else evaluate_handles(ctx, fit_bank, handles, downstream=downstream, r_handle=r_handle)
    diagnostics = []

    for index, handle in enumerate(handles):
        row = dict(handle)
        cal_result = cal_results[index]
        row["dcal_recovery"] = {
            "summary": cal_result["summary"],
            "sensitivity_score": float(cal_result["sensitivity_score"]),
            "invariance_score": float(cal_result["invariance_score"]),
            "passes": bool(cal_result["recovery_passed"]),
        }

        if downstream is None:
            empty = {
                "direct_output_matches_source": 0.0,
                "restored_Rmid_output_preserves_base": 0.0,
                "mean_output_effect_removed_fraction": 0.0,
                "passes": False,
            }
            row["dfit_restoration"] = {"split": "Dfit", **empty}
            row["dcal_restoration"] = {"split": "Dcal", **empty}
            row["restoration_direct_score"] = 0.0
            row["restoration_base_score"] = 0.0
            row["restoration_removed_score"] = 0.0
            row["restoration_passed_both"] = False
            row["mean_restoration_score"] = 0.0
            diagnostics.append(row)
            continue

        split_restorations = {}
        for split, result in (("Dfit", fit_results[index]), ("Dcal", cal_result)):
            restoration = dict(result["restoration"])
            restoration["passes"] = bool(
                restoration["direct_output_matches_source"] >= ctx.args.restoration_direct_threshold
                and restoration["restored_Rmid_output_preserves_base"] >= ctx.args.restoration_base_threshold
                and restoration["mean_output_effect_removed_fraction"] >= ctx.args.restoration_removed_threshold
            )
            split_restorations[split] = {"split": split, **restoration}

        row["dfit_restoration"] = split_restorations["Dfit"]
        row["dcal_restoration"] = split_restorations["Dcal"]
        row["restoration_direct_score"] = min(row["dfit_restoration"]["direct_output_matches_source"], row["dcal_restoration"]["direct_output_matches_source"])
        row["restoration_base_score"] = min(row["dfit_restoration"]["restored_Rmid_output_preserves_base"], row["dcal_restoration"]["restored_Rmid_output_preserves_base"])
        row["restoration_removed_score"] = min(row["dfit_restoration"]["mean_output_effect_removed_fraction"], row["dcal_restoration"]["mean_output_effect_removed_fraction"])
        row["restoration_passed_both"] = bool(
            row["restoration_direct_score"] >= ctx.args.restoration_direct_threshold
            and row["restoration_base_score"] >= ctx.args.restoration_base_threshold
            and row["restoration_removed_score"] >= ctx.args.restoration_removed_threshold
        )
        row["mean_restoration_score"] = mean([row["restoration_direct_score"], row["restoration_base_score"], row["restoration_removed_score"]])
        diagnostics.append(row)

    return diagnostics


def select_best_handle(ranked):
    # Recovery is required. Stop when no handle passes recovery on Dcal.
    valid = [handle for handle in ranked if handle["dcal_recovery"]["passes"]]
    if not valid:
        return None

    # Keep recovery-valid handles whose cosine is close to the best valid cosine.
    best_cosine = max(handle["signature_similarity"] for handle in valid)
    valid = [handle for handle in valid if best_cosine - handle["signature_similarity"] <= COSINE_TOLERANCE]

    def selection_key(handle):
        return discovery.handle_order(handle), handle["signature_mass"], -handle["k"], -abs(handle["strength"] - 1.0), handle["signature_similarity"]

    # If restoration passes, rank only the handles passing all criteria on both splits.
    restored = [handle for handle in valid if handle["restoration_passed_both"]]
    if restored:
        return max(restored, key=selection_key)

    # Use mean restoration only when every valid near-tied handle fails restoration.
    return max(valid, key=lambda handle: (handle["mean_restoration_score"], *selection_key(handle)))


def construct_candidate_handles(ctx, pool, strengths, variable, fit_bank, cal_bank, graded_fit, graded_cal, frozen_chain, r_handle=None):
    # Rank candidate handles, require recovery, and use restoration to resolve cosine near-ties.
    if not pool:
        print("  No candidate sites after variable classification", flush=True)
        return [], None, [], {"ranked": []}

    handles = build_handles(ctx, pool, strengths, variable)
    ranked, selector = rank_handles_by_signature(ctx, fit_bank, handles, frozen_chain)
    downstream = frozen_chain[0] if frozen_chain else None
    ranked = add_discovery_diagnostics(ctx, fit_bank, cal_bank, ranked, downstream, r_handle=r_handle)
    levels = []

    for size in range(1, ctx.args.max_handle_size + 1):
        size_handles = [handle for handle in ranked if handle["k"] == size]
        support_count = len({tuple(handle["site_ids"]) for handle in size_handles})
        levels.append({"k": size, "supports": support_count, "configurations": len(size_handles)})
        print(f"  k={size}: supports={support_count}, configurations={len(size_handles)}", flush=True)

    print("  Handle ranking:", flush=True)
    for index, handle in enumerate(ranked, 1):
        recovery = handle["dcal_recovery"]
        dfit_restoration = handle["dfit_restoration"]
        dcal_restoration = handle["dcal_restoration"]
        print(
            f"    {index}. sites={handle['site_ids']}, k={handle['k']}, strength={handle['strength']}, "
            f"cosine={handle['signature_similarity']:.4f}, mass={handle['signature_mass']:.4f}, "
            f"Dcal_score={float(recovery['summary']['score']):.4f}, "
            f"Dcal_sens={recovery['sensitivity_score']:.4f}, Dcal_inv={recovery['invariance_score']:.4f}, "
            f"Dcal_recovery={recovery['passes']}, "
            f"Dfit_direct={dfit_restoration['direct_output_matches_source']:.4f}, "
            f"Dfit_restored={dfit_restoration['restored_Rmid_output_preserves_base']:.4f}, "
            f"Dfit_removed={dfit_restoration['mean_output_effect_removed_fraction']:.4f}, "
            f"Dcal_direct={dcal_restoration['direct_output_matches_source']:.4f}, "
            f"Dcal_restored={dcal_restoration['restored_Rmid_output_preserves_base']:.4f}, "
            f"Dcal_removed={dcal_restoration['mean_output_effect_removed_fraction']:.4f}, "
            f"direct_score={handle['restoration_direct_score']:.4f}, "
            f"base_score={handle['restoration_base_score']:.4f}, "
            f"removed_score={handle['restoration_removed_score']:.4f}, "
            f"mean_restoration={handle['mean_restoration_score']:.4f}, "
            f"restoration_passed_both={handle['restoration_passed_both']}",
            flush=True,
        )

    chosen = select_best_handle(ranked)
    if chosen is None:
        print("  Stop: no handle passes Dcal recovery", flush=True)
        return ranked, None, levels, selector

    # Fit both decoders so the selected handle can be measured in later rounds.
    chosen = discovery.add_variable_metrics([chosen], graded_fit, graded_cal, ctx.args.graded_threshold)[0]
    chosen["variable"] = variable
    return ranked, chosen, levels, selector


# Step 3: causal graph construction
def name_nodes(handles, variable):
    # Name selected handles from early to late using computational position.
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
    # Check recovery, sensitivity, and invariance during final evaluation.
    sensitivity_passed = float(result["sensitivity_score"]) >= ctx.args.sensitivity_threshold
    invariance_passed = float(result["invariance_score"]) >= ctx.args.invariance_threshold
    return bool(result["recovery_passed"] and sensitivity_passed and invariance_passed)


def make_graph_edge(source, downstream_name, downstream_handle, evaluation_mode):
    # Store one edge in the progressively discovered linear chain.
    return {
        "source": source["name"],
        "downstream": downstream_name,
        "source_handle": source,
        "downstream_handle": downstream_handle,
        "evaluation_mode": evaluation_mode,
    }


def build_chain_edges(nodes, terminal_name, terminal_handle=None):
    # Connect each selected handle to the next downstream handle.
    edges = []
    for index, source in enumerate(nodes):
        if index + 1 < len(nodes):
            downstream = nodes[index + 1]
            edges.append(make_graph_edge(source, downstream["name"], downstream, "downstream"))
        elif terminal_handle is not None:
            edges.append(make_graph_edge(source, terminal_name, terminal_handle, "downstream"))
        else:
            edges.append(make_graph_edge(source, terminal_name, None, "output"))
    return edges


def expanded_edge_labels(edges):
    # Convert graph edges to compact text labels.
    return [f"{edge['source']} -> {edge['downstream']}" for edge in edges]


def save_graph_edges(edges):
    # Save graph structure without any discovery-time causal test fields.
    saved = []
    for edge in edges:
        row = {
            "edge": f"{edge['source']} -> {edge['downstream']}",
            "source": edge["source"],
            "downstream": edge["downstream"],
            "evaluation_mode": edge["evaluation_mode"],
            "source_handle": discovery.save_handle(edge["source_handle"]),
        }
        if edge["downstream_handle"] is not None:
            row["downstream_handle"] = discovery.save_handle(edge["downstream_handle"])
        saved.append(row)
    return saved


# Step 4: final graph evaluation
def evaluate_graph(ctx, bank, edges, r_handle=None):
    # Evaluate every selected edge on held-out Dte.
    results = []
    for edge in edges:
        source = edge["source_handle"]
        if edge["evaluation_mode"] == "output":
            result = evaluate_handles(ctx, bank, [source])[0]
        else:
            result = evaluate_handles(ctx, bank, [source], downstream=edge["downstream_handle"], r_handle=r_handle)[0]

        recovery_passed = downstream_passes(ctx, result)
        restoration = result.get("restoration")
        restoration_passed = None if restoration is None else bool(
            restoration["direct_output_matches_source"] >= ctx.args.restoration_direct_threshold
            and restoration["restored_Rmid_output_preserves_base"] >= ctx.args.restoration_base_threshold
            and restoration["mean_output_effect_removed_fraction"] >= ctx.args.restoration_removed_threshold
        )
        structure_passed = recovery_passed if restoration_passed is None else recovery_passed and restoration_passed

        row = dict(result)
        row["edge"] = f"{edge['source']} -> {edge['downstream']}"
        row["downstream_recovery_passed"] = recovery_passed
        row["heldout_restoration_passed"] = restoration_passed
        row["possible_bypass_to_Y"] = restoration_passed is False
        row["graph_structure_passed"] = bool(structure_passed)
        results.append(row)
    return results


def main():
    # Discover R and D progressively on Dfit/Dcal, then evaluate the complete graph on Dte.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = tuple(float(value) for value in args.strength_values.split(",") if value.strip())

    # Load model and candidate circuit.
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)
    site_lookup = {site.site_id: site for site in circuit.sites}
    ctx = discovery.Context(args, model, circuit.sites, site_lookup, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    # Build discovery banks. Dte is created only at final evaluation.
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
        # Search only sites upstream of the most recently frozen R handle.
        if r_downstream is None:
            r_sites = [site for site in ctx.sites if site.site_id not in r_frozen_sites]
        else:
            downstream_start = min(discovery.layer_order(site_id) for site_id in r_downstream["site_ids"])
            r_sites = [site for site in ctx.sites if discovery.layer_order(site.site_id) < downstream_start and site.site_id not in r_frozen_sites]
        if not r_sites:
            break

        # Screen sites, then rank real singleton/pair interventions using the whole frozen chain.
        frozen_chain = tuple(reversed(r_selected))
        print(f"[R round {round_index}/1] Candidate site selection", flush=True)
        r_ranking, r_pool = select_candidate_sites(ctx, coarse_fit, tuple(r_sites), "R", graded_fit, graded_cal, frozen_chain)

        print(f"[R round {round_index}/2] Candidate handle signature ranking", flush=True)
        r_handles, chosen, r_levels, r_selector = construct_candidate_handles(ctx, r_pool, strengths, "R", coarse_fit, coarse_cal, graded_fit, graded_cal, frozen_chain)

        round_result = {
            "round": round_index,
            "frozen_downstream_handle": None if r_downstream is None else discovery.save_handle(r_downstream),
            "frozen_chain": save_handles(frozen_chain),
            "candidate_site_selection": {"ranking": r_ranking, "candidate_pool_sites": r_pool},
            "candidate_handle_construction": {
                "search_levels": r_levels,
                "signature_ranking": save_handles(r_handles),
                "selector": r_selector,
                "cosine_tolerance": COSINE_TOLERANCE,
                "selection_rule": "require Dcal recovery; among cosine near-ties, prefer handles passing all restoration criteria on both Dfit and Dcal, then position, mass, smaller k, strength closest to 1, and cosine; if none passes restoration, use mean restoration first",
            },
            "chosen_handle": None,
        }

        if chosen is None:
            r_rounds.append(round_result)
            break

        # Stop when even the best handle no longer aligns with the abstract chain signature.
        if float(chosen["signature_similarity"]) < args.signature_threshold:
            round_result["stop_reason"] = f"best cosine {chosen['signature_similarity']:.4f} < threshold {args.signature_threshold:.4f}"
            r_rounds.append(round_result)
            print(f"[R round {round_index}] Stop: {round_result['stop_reason']}", flush=True)
            break

        # Freeze the selected handle and continue farther upstream.
        round_result["chosen_handle"] = discovery.save_handle(chosen)
        r_rounds.append(round_result)
        r_selected.append(chosen)
        r_frozen_sites.update(chosen["site_ids"])
        r_downstream = chosen
        print(f"[R round {round_index}] Chosen: {chosen['site_ids']}, strength={chosen['strength']}, cosine={chosen['signature_similarity']:.4f}, mass={chosen['signature_mass']:.4f}, Dcal_recovery={chosen['dcal_recovery']['passes']}, mean_restoration={chosen['mean_restoration_score']:.4f}, restoration_passed_both={chosen['restoration_passed_both']}", flush=True)
        round_index += 1

    if not r_selected:
        raise RuntimeError("No R handle passed the signature threshold")

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
        # Search D only upstream of the most recently frozen downstream handle.
        downstream_start = min(discovery.layer_order(site_id) for site_id in d_downstream["site_ids"])
        d_sites = [site for site in ctx.sites if discovery.layer_order(site.site_id) < downstream_start and site.site_id not in d_frozen_sites]
        if not d_sites:
            break

        # The D signature includes frozen D handles, the complete R chain, and Y.
        frozen_chain = tuple(reversed(d_selected)) + tuple(r_nodes)
        print(f"[D round {round_index}/1] Candidate site selection", flush=True)
        d_ranking, d_pool = select_candidate_sites(ctx, graded_fit, tuple(d_sites), "D", graded_fit, graded_cal, frozen_chain)

        print(f"[D round {round_index}/2] Candidate handle signature ranking", flush=True)
        d_handles, chosen, d_levels, d_selector = construct_candidate_handles(ctx, d_pool, strengths, "D", graded_fit, graded_cal, graded_fit, graded_cal, frozen_chain, r_handle=r_early)

        round_result = {
            "round": round_index,
            "frozen_downstream_handle": discovery.save_handle(d_downstream),
            "frozen_chain": save_handles(frozen_chain),
            "candidate_site_selection": {"ranking": d_ranking, "candidate_pool_sites": d_pool},
            "candidate_handle_construction": {
                "search_levels": d_levels,
                "signature_ranking": save_handles(d_handles),
                "selector": d_selector,
                "cosine_tolerance": COSINE_TOLERANCE,
                "selection_rule": "require Dcal recovery; among cosine near-ties, prefer handles passing all restoration criteria on both Dfit and Dcal, then position, mass, smaller k, strength closest to 1, and cosine; if none passes restoration, use mean restoration first",
            },
            "chosen_handle": None,
        }

        if chosen is None:
            d_rounds.append(round_result)
            break

        # Use the same signature stopping rule as R.
        if float(chosen["signature_similarity"]) < args.signature_threshold:
            round_result["stop_reason"] = f"best cosine {chosen['signature_similarity']:.4f} < threshold {args.signature_threshold:.4f}"
            d_rounds.append(round_result)
            print(f"[D round {round_index}] Stop: {round_result['stop_reason']}", flush=True)
            break

        # Freeze the selected D handle and continue upstream.
        round_result["chosen_handle"] = discovery.save_handle(chosen)
        d_rounds.append(round_result)
        d_selected.append(chosen)
        d_frozen_sites.update(chosen["site_ids"])
        d_downstream = chosen
        print(f"[D round {round_index}] Chosen: {chosen['site_ids']}, strength={chosen['strength']}, cosine={chosen['signature_similarity']:.4f}, mass={chosen['signature_mass']:.4f}, Dcal_recovery={chosen['dcal_recovery']['passes']}, mean_restoration={chosen['mean_restoration_score']:.4f}, restoration_passed_both={chosen['restoration_passed_both']}", flush=True)
        round_index += 1

    # Build D graph if discovered. Otherwise continue to evaluate the R graph.
    if d_selected:
        print("[D/3] Causal graph construction", flush=True)
        d_nodes = name_nodes(d_selected, "D")
        d_edges = build_chain_edges(d_nodes, r_early["name"], terminal_handle=r_early)
    else:
        print("[D/3] No D chain discovered", flush=True)
        d_nodes = []
        d_edges = []

    # Causal tests are run only after the complete graph has been discovered.
    print("[4] Final graph evaluation on Dte", flush=True)
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
    config["cosine_tolerance"] = COSINE_TOLERANCE
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