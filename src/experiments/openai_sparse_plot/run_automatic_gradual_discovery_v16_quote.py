from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import torch

from . import run_automatic_gradual_discovery_v10 as discovery
from .ablate_rediscover import build_quote_pairs, build_quote_rediscovery_bank
from .plot_matching import cost_matrix
from .runtime import quote_token_ids


EXPERIMENT_NAME = "automatic_gradual_discovery_quote_v16"


def parse_args():
    # Configure signature-only selection, Dcal causal diagnostics, and held-out evaluation.
    parser = argparse.ArgumentParser(description="Signature-only progressive discovery for the closing-quote task.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/quote_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"outputs/{EXPERIMENT_NAME}"))
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-handle-size", type=int, default=2)
    parser.add_argument("--strength-values", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--signature-threshold", type=float, default=0.9)
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.9)
    parser.add_argument("--invariance-threshold", type=float, default=0.9)
    parser.add_argument("--restoration-direct-threshold", type=float, default=0.9)
    parser.add_argument("--restoration-base-threshold", type=float, default=0.8)
    parser.add_argument("--restoration-removed-threshold", type=float, default=0.5)
    parser.add_argument("--token-position", choices=("opening", "final"), default="opening")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def set_token_position(examples, token_position):
    # Use one task-defined token position for every candidate in this run.
    if token_position == "opening":
        return examples
    return tuple(replace(example, patch_position=len(example.token_ids) - 1) for example in examples)


def save_handle(handle):
    # Convert the U readout object into a JSON-safe dictionary.
    saved = {key: value for key, value in handle.items() if key != "u_readout"}
    if "u_readout" in handle:
        saved["u_readout"] = handle["u_readout"].to_dict()
    return saved


def save_handles(handles):
    # Convert a list of handles or evaluation results to JSON-safe dictionaries.
    return [save_handle(handle) for handle in handles]


def short_handle(handle):
    # Keep the compact handle information used by the summary file.
    return {"sites": handle["site_ids"], "weights": handle["weights"], "strength": handle["strength"]}


# Signature matching
def match_signatures_cosine(abstract, neural):
    # Compare each neural signature with the abstract signature using cosine similarity.
    candidate_ids = tuple(neural)
    abstract_tensor = torch.tensor([list(abstract)], dtype=torch.float32)
    neural_tensor = torch.tensor([list(neural[candidate_id]) for candidate_id in candidate_ids], dtype=torch.float32)
    costs = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    similarities = 1.0 - costs[0]
    weights = similarities.clamp_min(0.0)

    # Convert positive cosine similarities into normalized masses.
    if float(weights.sum()) <= 0.0:
        weights = torch.softmax(similarities, dim=0)
    else:
        weights = weights / weights.sum()

    ranked = []
    for index, candidate_id in enumerate(candidate_ids):
        ranked.append({
            "candidate_id": candidate_id,
            "weight": float(weights[index]),
            "similarity": float(similarities[index]),
            "cost": float(costs[0, index]),
        })

    ranked.sort(key=lambda row: (-row["similarity"], row["candidate_id"]))
    return {"cost_mode": "raw_cosine", "matching": "direct_cosine_similarity", "ranked": ranked}


def abstract_u_signature(bank):
    # Build the expected source-minus-base change of unmatched quote type U.
    return tuple(
        float(bank.by_id[pair.source_id].variable_value) - float(bank.by_id[pair.base_id].variable_value)
        for pair in bank.pairs
    )


def build_chain_signatures(ctx, bank, handles, frozen_chain):
    # Concatenate each candidate's effect on every frozen U handle and on Y.
    neural = {handle["handle_id"]: [] for handle in handles}
    abstract = []

    # Collect the frozen sites that must be measured during candidate interventions.
    downstream_sites = []
    seen_sites = set()
    for frozen in frozen_chain:
        for site in discovery.get_downstream_sites(ctx, frozen):
            if site.site_id not in seen_sites:
                downstream_sites.append(site)
                seen_sites.add(site.site_id)

    # Patch every candidate once and measure the complete frozen chain plus Y.
    if downstream_sites:
        output_margins, measured_values = discovery.run_handles_and_measure(ctx, bank, handles, tuple(downstream_sites))
    else:
        output_margins = discovery.run_handles(ctx, bank, handles)
        measured_values = None

    measured_site_ids = tuple(site.site_id for site in downstream_sites)

    def append_block(abstract_block, neural_block):
        # Normalize each chain node separately so every node contributes equally.
        abstract_tensor = torch.tensor(abstract_block, dtype=torch.float32)
        abstract_norm = float(torch.linalg.vector_norm(abstract_tensor))
        if abstract_norm > 0.0:
            abstract_tensor = abstract_tensor / abstract_norm
        abstract.extend(float(value) for value in abstract_tensor)

        for candidate_id, values in neural_block.items():
            values_tensor = torch.tensor(values, dtype=torch.float32)
            values_norm = float(torch.linalg.vector_norm(values_tensor))
            if values_norm > 0.0:
                values_tensor = values_tensor / values_norm
            neural[candidate_id].extend(float(value) for value in values_tensor)

    # Add frozen U handles from the nearest downstream handle toward the output.
    for frozen in frozen_chain:
        frozen_site_ids = tuple(frozen["weights"])
        indices = [measured_site_ids.index(site_id) for site_id in frozen_site_ids]
        orientation = float(frozen["u_readout"].orientation)
        neural_block = {}

        for handle_index, handle in enumerate(handles):
            values = measured_values[handle_index][..., indices]
            combined_values = discovery.combine_downstream_values(values, frozen_site_ids, frozen["weights"])
            signature = []
            for pair_index, pair in enumerate(bank.pairs):
                base_value = discovery.handle_value(bank.runs[pair.base_id], frozen["weights"])
                signature.append(orientation * (float(combined_values[pair_index]) - float(base_value)))
            neural_block[handle["handle_id"]] = tuple(signature)

        append_block(abstract_u_signature(bank), neural_block)

    # Always include the candidate's output-margin effect as the final block.
    output_block = {}
    for handle_index, handle in enumerate(handles):
        output_signature = []
        for pair_index, pair in enumerate(bank.pairs):
            patched_margin = float(output_margins[handle_index, pair_index])
            base_margin = float(bank.runs[pair.base_id].class_margin)
            output_signature.append(patched_margin - base_margin)
        output_block[handle["handle_id"]] = tuple(output_signature)
    append_block(abstract_u_signature(bank), output_block)

    return tuple(abstract), {candidate_id: tuple(values) for candidate_id, values in neural.items()}


def rank_handles_by_signature(ctx, bank, handles, frozen_chain):
    # Assign every handle configuration a cosine score and normalized mass.
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

    # Prefer mass, then concise handles, then later position, then strength near one.
    ranked.sort(
        key=lambda row: (
            row["signature_mass"],
            -row["k"],
            discovery.handle_order(row),
            -abs(row["strength"] - 1.0),
        ),
        reverse=True,
    )
    return ranked, selector


# Step 1: candidate site selection
def rank_sites(ctx, bank, sites, frozen_chain):
    # Screen every eligible site as a strength-one singleton.
    singletons = [
        {"handle_id": site.site_id, "site_ids": [site.site_id], "weights": {site.site_id: 1.0}, "k": 1, "strength": 1.0, "variable": "U"}
        for site in sites
    ]
    ranked_handles, selector = rank_handles_by_signature(ctx, bank, singletons, frozen_chain)
    ranked_sites = [
        {
            "site_id": handle["site_ids"][0],
            "weight": handle["signature_mass"],
            "similarity": handle["signature_similarity"],
            "cost": handle["signature_cost"],
        }
        for handle in ranked_handles
    ]
    return {"ranked_sites": ranked_sites, "selector": selector, "chain_depth": len(frozen_chain)}


def select_candidate_sites(ctx, bank, sites, frozen_chain):
    # Rank all eligible sites, then keep the requested Top-K candidate pool.
    ranking = rank_sites(ctx, bank, sites, frozen_chain)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)

    print("  Site signature ranking:", flush=True)
    for index, row in enumerate(ranking["ranked_sites"][:ctx.args.candidate_pool_size], 1):
        print(f"    {index}. site={row['site_id']}, cosine={row['similarity']:.4f}, mass={row['weight']:.4f}", flush=True)
    return ranking, pool


# Step 2: candidate handle construction and signature ranking
def build_handles(ctx, pool, strengths):
    # Build every singleton/pair/... support and instantiate every strength.
    site_rows = {str(row["site_id"]): row for row in pool}
    handles = []
    support_index = 1

    for size in range(1, ctx.args.max_handle_size + 1):
        for support in combinations(site_rows, size):
            total_mass = sum(float(site_rows[site_id]["weight"]) for site_id in support)
            weights = {}
            for site_id in support:
                weights[site_id] = float(site_rows[site_id]["weight"]) / total_mass if total_mass > 0.0 else 1.0 / size

            for strength in strengths:
                handles.append({
                    "handle_id": f"k{size}_{support_index}_s{strength:g}",
                    "site_ids": list(support),
                    "weights": weights,
                    "k": size,
                    "strength": float(strength),
                    "variable": "U",
                    "site_mass": total_mass,
                })
            support_index += 1
    return handles


def construct_candidate_handles(ctx, pool, strengths, signature_bank, cal_bank, frozen_chain):
    # Rank handles by signature, then report Dcal recovery/restoration without using them for selection.
    if not pool:
        print("  No candidate sites", flush=True)
        return [], None, [], {"ranked": []}

    handles = build_handles(ctx, pool, strengths)
    ranked, selector = rank_handles_by_signature(ctx, signature_bank, handles, frozen_chain)

    # Round 1 tests recovery at Y. Later rounds test recovery and restoration through the immediate downstream handle.
    # These diagnostics are attached after signature ranking and never change the ranking or chosen handle.
    downstream_handle = frozen_chain[0] if frozen_chain else None
    downstream_name = "Y" if downstream_handle is None else downstream_handle["site_ids"]
    print(f"  Dcal diagnostics downstream: {downstream_name}", flush=True)
    ranked = add_dcal_diagnostics(ctx, cal_bank, ranked, downstream_handle)

    levels = []

    for size in range(1, ctx.args.max_handle_size + 1):
        size_handles = [handle for handle in ranked if handle["k"] == size]
        support_count = len({tuple(handle["site_ids"]) for handle in size_handles})
        levels.append({"k": size, "supports": support_count, "configurations": len(size_handles)})
        print(f"  k={size}: supports={support_count}, configurations={len(size_handles)}", flush=True)

    print("  Handle mass ranking:", flush=True)
    for index, handle in enumerate(ranked, 1):
        recovery = handle["dcal_recovery"]
        restoration = handle.get("dcal_restoration")
        diagnostic_text = (
            f", Dcal_recovery_score={recovery['score']:.4f}, "
            f"Dcal_sens={recovery['sensitivity_score']:.4f}, "
            f"Dcal_inv={recovery['invariance_score']:.4f}, "
            f"Dcal_recovery={recovery['passes']}"
        )
        if restoration is not None:
            diagnostic_text += (
                f", Dcal_direct={restoration['direct_output_matches_source']:.4f}, "
                f"Dcal_restored={restoration['restored_output_preserves_base']:.4f}, "
                f"Dcal_removed={restoration['mean_output_effect_removed_fraction']:.4f}, "
                f"Dcal_restoration={restoration['passes']}"
            )
        print(
            f"    {index}. sites={handle['site_ids']}, k={handle['k']}, strength={handle['strength']}, "
            f"cosine={handle['signature_similarity']:.4f}, mass={handle['signature_mass']:.4f}{diagnostic_text}",
            flush=True,
        )

    if not ranked:
        return [], None, levels, selector
    return ranked, dict(ranked[0]), levels, selector


# Step 3: causal graph construction
def name_nodes(handles):
    # Name the selected U handles from early to late computational position.
    nodes = [dict(handle) for handle in handles]
    nodes.sort(key=discovery.handle_order)

    if len(nodes) == 1:
        nodes[0]["name"] = "U"
        return nodes

    nodes[0]["name"] = "U_early"
    nodes[-1]["name"] = "U_late"
    for index, node in enumerate(reversed(nodes[1:-1]), 1):
        node["name"] = f"U_mid_{index}"
    return nodes


def make_graph_edge(source, downstream_name, downstream_handle, evaluation_mode):
    # Store graph structure without using any discovery-time causal test.
    return {
        "source": source["name"],
        "downstream": downstream_name,
        "source_handle": source,
        "downstream_handle": downstream_handle,
        "evaluation_mode": evaluation_mode,
    }


def build_chain_edges(nodes):
    # Connect every U handle to the next downstream handle and U_late to Y.
    edges = []
    for index, source in enumerate(nodes):
        if index + 1 < len(nodes):
            downstream = nodes[index + 1]
            edges.append(make_graph_edge(source, downstream["name"], downstream, "downstream"))
        else:
            edges.append(make_graph_edge(source, "Y", None, "output"))
    return edges


def expanded_edge_labels(edges):
    # Convert graph edges into compact text labels.
    return [f"{edge['source']} -> {edge['downstream']}" for edge in edges]


def save_graph_edges(edges):
    # Save graph structure and both endpoint handles.
    saved = []
    for edge in edges:
        row = {
            "edge": f"{edge['source']} -> {edge['downstream']}",
            "source": edge["source"],
            "downstream": edge["downstream"],
            "evaluation_mode": edge["evaluation_mode"],
            "source_handle": save_handle(edge["source_handle"]),
        }
        if edge["downstream_handle"] is not None:
            row["downstream_handle"] = save_handle(edge["downstream_handle"])
        saved.append(row)
    return saved


# Step 4: final graph evaluation
def mean(values):
    # Return the arithmetic mean of a nonempty list.
    return float(sum(float(value) for value in values) / len(values))


def output_recovery_summary(bank, patched_margins):
    # Evaluate U recovery at Y for different-U and same-U intervention pairs.
    by_relation = defaultdict(list)
    for pair, margin in zip(bank.pairs, patched_margins):
        base_u = int(bank.by_id[pair.base_id].variable_value)
        source_u = int(bank.by_id[pair.source_id].variable_value)
        expected_u = source_u if source_u != base_u else base_u
        predicted_u = 1 if float(margin) > 0.0 else -1
        by_relation[pair.relation].append(predicted_u == expected_u)

    rates = {relation: mean(values) for relation, values in sorted(by_relation.items())}
    sensitivity = float(rates.get("different_variable", 0.0))
    invariance = float(rates.get("same_variable", 0.0))
    return {
        "rates": rates,
        "score": mean(list(rates.values())),
        "sensitivity_score": sensitivity,
        "invariance_score": invariance,
    }


def downstream_recovery_summary(bank, patched_margins, patched_values, downstream_handle):
    # Evaluate whether the source U handle correctly controls a frozen downstream U handle and Y.
    by_relation = defaultdict(list)
    sensitivity_rows = []
    invariance_rows = []

    for index, pair in enumerate(bank.pairs):
        base_u = int(bank.by_id[pair.base_id].variable_value)
        source_u = int(bank.by_id[pair.source_id].variable_value)
        expected_u = source_u if source_u != base_u else base_u
        output_u = 1 if float(patched_margins[index]) > 0.0 else -1
        downstream_u = downstream_handle["u_readout"].predict(float(patched_values[index]))
        row = {"output_correct": output_u == expected_u, "downstream_correct": downstream_u == expected_u}
        by_relation[pair.relation].append(row)

        if source_u != base_u:
            sensitivity_rows.append(row)
        else:
            invariance_rows.append(row)

    relation_metrics = {
        relation: {
            "output_correct": mean([row["output_correct"] for row in rows]),
            "downstream_correct": mean([row["downstream_correct"] for row in rows]),
        }
        for relation, rows in sorted(by_relation.items())
    }
    balanced_blocks = {
        "sensitivity_output": mean([row["output_correct"] for row in sensitivity_rows]),
        "sensitivity_downstream": mean([row["downstream_correct"] for row in sensitivity_rows]),
        "invariance_output": mean([row["output_correct"] for row in invariance_rows]),
        "invariance_downstream": mean([row["downstream_correct"] for row in invariance_rows]),
    }
    return {
        "relations": relation_metrics,
        "balanced_blocks": balanced_blocks,
        "score": mean(list(balanced_blocks.values())),
        "sensitivity_score": min(balanced_blocks["sensitivity_output"], balanced_blocks["sensitivity_downstream"]),
        "invariance_score": min(balanced_blocks["invariance_output"], balanced_blocks["invariance_downstream"]),
    }


def restoration_summary(ctx, bank, direct_margins, restored_margins):
    # Test whether restoring the downstream U handle removes the source handle's output effect.
    different_indices = [
        index
        for index, pair in enumerate(bank.pairs)
        if bank.by_id[pair.base_id].variable_value != bank.by_id[pair.source_id].variable_value
    ]
    direct_source = []
    restored_base = []
    removed_fractions = []

    for index in different_indices:
        pair = bank.pairs[index]
        base_u = int(bank.by_id[pair.base_id].variable_value)
        source_u = int(bank.by_id[pair.source_id].variable_value)
        direct_u = 1 if float(direct_margins[index]) > 0.0 else -1
        restored_u = 1 if float(restored_margins[index]) > 0.0 else -1
        direct_source.append(direct_u == source_u)
        restored_base.append(restored_u == base_u)

        base_margin = float(bank.runs[pair.base_id].class_margin)
        direct_effect = abs(float(direct_margins[index]) - base_margin)
        remaining_effect = abs(float(restored_margins[index]) - base_margin)
        if direct_effect > 1e-8:
            removed_fractions.append(1.0 - remaining_effect / direct_effect)

    direct_rate = mean(direct_source)
    restored_rate = mean(restored_base)
    removed_fraction = mean(removed_fractions) if removed_fractions else 0.0
    passed = (
        direct_rate >= ctx.args.restoration_direct_threshold
        and restored_rate >= ctx.args.restoration_base_threshold
        and removed_fraction >= ctx.args.restoration_removed_threshold
    )
    return {
        "different_U_records": len(different_indices),
        "direct_output_matches_source": direct_rate,
        "restored_output_preserves_base": restored_rate,
        "mean_output_effect_removed_fraction": removed_fraction,
        "passes": bool(passed),
    }


def add_dcal_diagnostics(ctx, cal_bank, handles, downstream_handle):
    # Report recovery for every round and restoration whenever a frozen downstream handle exists.
    if not handles:
        return handles

    if downstream_handle is None:
        direct_margins = discovery.run_handles(ctx, cal_bank, handles)
        measured_values = None
        downstream_site_ids = ()
        restored_margins = None
    else:
        downstream_sites = discovery.get_downstream_sites(ctx, downstream_handle)
        downstream_site_ids = tuple(site.site_id for site in downstream_sites)
        direct_margins, measured_values = discovery.run_handles_and_measure(ctx, cal_bank, handles, downstream_sites)
        restored_margins, _ = discovery.run_handles_and_measure(
            ctx,
            cal_bank,
            handles,
            downstream_sites,
            tuple(downstream_handle["weights"]),
        )

    # Attach diagnostics without sorting again, so selection remains signature-only.
    results = []
    for index, handle in enumerate(handles):
        row = dict(handle)
        if downstream_handle is None:
            recovery = output_recovery_summary(cal_bank, direct_margins[index])
            evaluation_mode = "output"
        else:
            patched_values = discovery.combine_downstream_values(measured_values[index], downstream_site_ids, downstream_handle["weights"])
            recovery = downstream_recovery_summary(cal_bank, direct_margins[index], patched_values, downstream_handle)
            evaluation_mode = "downstream"

        recovery_passed = (
            recovery["sensitivity_score"] >= ctx.args.sensitivity_threshold
            and recovery["invariance_score"] >= ctx.args.invariance_threshold
        )
        row["dcal_recovery"] = {
            "split": "Dcal",
            "evaluation_mode": evaluation_mode,
            "downstream_site_ids": [] if downstream_handle is None else list(downstream_handle["site_ids"]),
            **recovery,
            "passes": bool(recovery_passed),
        }

        if downstream_handle is not None:
            restoration = restoration_summary(ctx, cal_bank, direct_margins[index], restored_margins[index])
            row["dcal_restoration"] = {
                "split": "Dcal",
                "downstream_site_ids": list(downstream_handle["site_ids"]),
                **restoration,
            }
        results.append(row)
    return results


def evaluate_output_edge(ctx, bank, source):
    # Test recovery for the terminal U_late -> Y edge; restoration is not applicable.
    patched_margins = discovery.run_handles(ctx, bank, [source])[0]
    summary = output_recovery_summary(bank, patched_margins)
    recovery_passed = (
        summary["sensitivity_score"] >= ctx.args.sensitivity_threshold
        and summary["invariance_score"] >= ctx.args.invariance_threshold
    )
    return {
        **source,
        "summary": summary,
        "sensitivity_score": summary["sensitivity_score"],
        "invariance_score": summary["invariance_score"],
        "recovery_passed": bool(recovery_passed),
        "restoration": None,
        "restoration_passed": None,
    }


def evaluate_downstream_edge(ctx, bank, source, downstream):
    # Test recovery and restoration for one internal U -> U edge.
    downstream_sites = discovery.get_downstream_sites(ctx, downstream)
    downstream_site_ids = tuple(site.site_id for site in downstream_sites)
    direct_margins, measured_values = discovery.run_handles_and_measure(ctx, bank, [source], downstream_sites)
    restored_margins, _ = discovery.run_handles_and_measure(ctx, bank, [source], downstream_sites, tuple(downstream["weights"]))
    patched_values = discovery.combine_downstream_values(measured_values[0], downstream_site_ids, downstream["weights"])
    summary = downstream_recovery_summary(bank, direct_margins[0], patched_values, downstream)
    restoration = restoration_summary(ctx, bank, direct_margins[0], restored_margins[0])
    recovery_passed = (
        summary["sensitivity_score"] >= ctx.args.sensitivity_threshold
        and summary["invariance_score"] >= ctx.args.invariance_threshold
    )
    return {
        **source,
        "summary": summary,
        "sensitivity_score": summary["sensitivity_score"],
        "invariance_score": summary["invariance_score"],
        "recovery_passed": bool(recovery_passed),
        "restoration": restoration,
        "restoration_passed": bool(restoration["passes"]),
    }


def evaluate_graph(ctx, bank, edges):
    # Run recovery and restoration for the first time on held-out Dte.
    results = []
    for edge in edges:
        if edge["evaluation_mode"] == "output":
            result = evaluate_output_edge(ctx, bank, edge["source_handle"])
        else:
            result = evaluate_downstream_edge(ctx, bank, edge["source_handle"], edge["downstream_handle"])

        recovery_passed = bool(result["recovery_passed"])
        restoration_passed = result["restoration_passed"]
        structure_passed = recovery_passed if restoration_passed is None else recovery_passed and bool(restoration_passed)
        row = dict(result)
        row["edge"] = f"{edge['source']} -> {edge['downstream']}"
        row["heldout_recovery_passed"] = recovery_passed
        row["heldout_restoration_passed"] = restoration_passed
        row["possible_bypass_to_Y"] = edge["downstream"] != "Y" and restoration_passed is False
        row["graph_structure_passed"] = bool(structure_passed)
        results.append(row)
    return results


def main():
    # Discover U by signature, report Dcal recovery/restoration each round, then certify the graph on Dte.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = tuple(float(value) for value in args.strength_values.split(",") if value.strip())

    # Load the csp_yolo1 model and all 64 localized quote candidates.
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=64)
    model, model_info = discovery.load_sparse_gpt_model(
        model_name="csp_yolo1",
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=True,
        grad_checkpointing=False,
    )
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)
    site_lookup = {site.site_id: site for site in circuit.sites}
    output_tokens = quote_token_ids(encoder)
    device = "cuda" if args.cuda else "cpu"
    ctx = discovery.Context(args, model, circuit.sites, site_lookup, output_tokens["single"], output_tokens["double"], device)

    # Build disjoint Dfit, Dcal, and Dte examples; only collect Dfit/Dcal now.
    examples = build_quote_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    # examples = set_token_position(examples, args.token_position)
    fit_pairs = build_quote_pairs(examples, split="Dfit", records_per_relation=100)
    cal_pairs = build_quote_pairs(examples, split="Dcal", records_per_relation=100)
    fit_bank = discovery.make_bank(ctx, examples, fit_pairs, "Dfit")
    cal_bank = discovery.make_bank(ctx, examples, cal_pairs, "Dcal")

    # Discover U progressively: Y <- U_late <- U_mid <- ... <- U_early.
    u_rounds = []
    u_selected = []
    frozen_site_ids = set()
    downstream_handle = None
    round_index = 1

    while True:
        # Search only computationally upstream sites that have not been frozen.
        if downstream_handle is None:
            eligible_sites = [site for site in ctx.sites if site.site_id not in frozen_site_ids]
        else:
            downstream_order = discovery.handle_order(downstream_handle)
            eligible_sites = [
                site
                for site in ctx.sites
                if discovery.layer_order(site.site_id) < downstream_order and site.site_id not in frozen_site_ids
            ]
        if not eligible_sites:
            break

        # Rank sites and handles using signatures over the complete frozen chain.
        frozen_chain = tuple(reversed(u_selected))
        print(f"[U round {round_index}/1] Candidate site selection", flush=True)
        site_ranking, candidate_pool = select_candidate_sites(ctx, fit_bank, tuple(eligible_sites), frozen_chain)

        print(f"[U round {round_index}/2] Candidate handle signature ranking", flush=True)
        ranked_handles, chosen, search_levels, handle_selector = construct_candidate_handles(ctx, candidate_pool, strengths, fit_bank, cal_bank, frozen_chain)

        round_result = {
            "round": round_index,
            "frozen_downstream_handle": None if downstream_handle is None else save_handle(downstream_handle),
            "frozen_chain": save_handles(frozen_chain),
            "candidate_site_selection": {"ranking": site_ranking, "candidate_pool_sites": candidate_pool},
            "candidate_handle_construction": {
                "search_levels": search_levels,
                "signature_ranking": save_handles(ranked_handles),
                "selector": handle_selector,
                "selection_rule": "highest signature mass",
                "dcal_recovery_used_for_selection": False,
                "dcal_restoration_used_for_selection": False,
            },
            "chosen_handle": None,
        }

        if chosen is None:
            u_rounds.append(round_result)
            break

        # Stop before freezing a candidate whose chain cosine is too small.
        if float(chosen["signature_similarity"]) < args.signature_threshold:
            round_result["stop_reason"] = f"best cosine {chosen['signature_similarity']:.4f} < threshold {args.signature_threshold:.4f}"
            u_rounds.append(round_result)
            print(f"[U round {round_index}] Stop: {round_result['stop_reason']}", flush=True)
            break

        # Fit a binary U readout only so later rounds can measure this frozen handle.
        u_readout, u_accuracy = discovery.fit_r(fit_bank, cal_bank, chosen["weights"])
        chosen["u_readout"] = u_readout
        chosen["u_accuracy"] = u_accuracy
        chosen["variable"] = "U"

        # Freeze the selected handle and continue farther upstream.
        round_result["chosen_handle"] = save_handle(chosen)
        u_rounds.append(round_result)
        u_selected.append(chosen)
        frozen_site_ids.update(chosen["site_ids"])
        downstream_handle = chosen
        recovery = chosen["dcal_recovery"]
        restoration = chosen.get("dcal_restoration")
        diagnostic_text = (
            f", Dcal_recovery_score={recovery['score']:.4f}, "
            f"Dcal_sens={recovery['sensitivity_score']:.4f}, "
            f"Dcal_inv={recovery['invariance_score']:.4f}, "
            f"Dcal_recovery={recovery['passes']}"
        )
        if restoration is not None:
            diagnostic_text += (
                f", Dcal_restored={restoration['restored_output_preserves_base']:.4f}, "
                f"Dcal_removed={restoration['mean_output_effect_removed_fraction']:.4f}, "
                f"Dcal_restoration={restoration['passes']}"
            )
        print(
            f"[U round {round_index}] Chosen: {chosen['site_ids']}, strength={chosen['strength']}, "
            f"cosine={chosen['signature_similarity']:.4f}, mass={chosen['signature_mass']:.4f}, "
            f"readout_fit={u_accuracy['Dfit']:.4f}, readout_cal={u_accuracy['Dcal']:.4f}{diagnostic_text}",
            flush=True,
        )
        round_index += 1

    if not u_selected:
        raise RuntimeError("No U handle passed the signature threshold")

    # Construct the discovered linear U chain without running causal tests.
    print("[U/3] Causal graph construction", flush=True)
    u_nodes = name_nodes(u_selected)
    u_edges = build_chain_edges(u_nodes)

    # Build Dte only after discovery, then evaluate every discovered edge.
    print("[4] Final graph evaluation on Dte", flush=True)
    test_pairs = build_quote_pairs(examples, split="Dte", records_per_relation=100)
    test_bank = discovery.make_bank(ctx, examples, test_pairs, "Dte")
    test_results = evaluate_graph(ctx, test_bank, u_edges)
    passed = all(row["graph_structure_passed"] for row in test_results)
    final_model = "; ".join(expanded_edge_labels(u_edges))

    # Save detailed discovery records and a compact graph summary.
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    detailed_name = f"{EXPERIMENT_NAME}_detailed.json"
    summary_name = f"{EXPERIMENT_NAME}_summary.json"
    detailed = {
        "experiment": EXPERIMENT_NAME,
        "config": config,
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "U": {
            "progressive_rounds": u_rounds,
            "causal_graph_construction": {
                "nodes": save_handles(u_nodes),
                "edges": save_graph_edges(u_edges),
                "expanded_edges": expanded_edge_labels(u_edges),
            },
        },
        "final_graph_evaluation": {"split": "Dte", "U_edges": save_handles(test_results), "passed": passed},
        "final_model": final_model,
        "passed": passed,
    }
    handles = {node["name"]: short_handle(node) for node in u_nodes}
    summary = {
        "final_model": final_model,
        "passed": passed,
        "handles": handles,
        "U_edges": save_graph_edges(u_edges),
        "detailed_output": detailed_name,
    }

    detailed_path = args.out_dir / detailed_name
    summary_path = args.out_dir / summary_name
    discovery.atomic_json(detailed_path, detailed)
    discovery.atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()