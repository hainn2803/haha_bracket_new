from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


def parse_args():
    parser = argparse.ArgumentParser(description="Ordered downstream graph discovery with bypass edges.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("data/bracket_circuit_nodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/automatic_gradual_discovery_v18"))
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--max-handle-size", type=int, default=3)
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


def save_handles(handles):
    saved = []
    for handle in handles:
        saved.append(discovery.save_handle(handle))
    return saved


def rank_pool(ctx, bank, sites, downstream=None):
    ranking = discovery.rank_sites(ctx, bank, sites, downstream_handle=downstream)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)
    return ranking, pool


def build_handles(ctx, supports, pool, strengths, variable, fit_bank, cal_bank):
    site_rows = {}
    for row in pool:
        site_rows[str(row["site_id"])] = row

    base_handles = []
    for index, support in enumerate(supports, 1):
        total_mass = 0.0
        for site_id in support:
            total_mass += float(site_rows[site_id]["weight"])

        weights = {}
        for site_id in support:
            if total_mass > 0.0:
                weights[site_id] = float(site_rows[site_id]["weight"]) / total_mass
            else:
                weights[site_id] = 1.0 / len(support)

        base_handles.append({"handle_id": f"k{len(support)}_{index}", "site_ids": list(support), "weights": weights, "k": len(support), "strength": 1.0, "ot_mass": total_mass})

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


def evaluate_handles(ctx, bank, handles, downstream=None, r_handle=None, restore_handle=None, self_downstream=False):
    if self_downstream:
        results = []
        for handle in handles:
            result = discovery.evaluate_handles(ctx, bank, [handle], downstream_handle=handle, r_handle=r_handle, restore_handle=restore_handle)[0]
            results.append(result)
        return results
    return discovery.evaluate_handles(ctx, bank, handles, downstream_handle=downstream, r_handle=r_handle, restore_handle=restore_handle)


def split_supports(supports, valid_configurations):
    valid_by_support = {}
    for row in valid_configurations:
        support = tuple(row["site_ids"])
        if support not in valid_by_support:
            valid_by_support[support] = []
        valid_by_support[support].append(row)

    valid_handles = []
    for rows in valid_by_support.values():
        valid_handles.append(max(rows, key=discovery.handle_selection_key))

    failed = []
    for support in supports:
        if support not in valid_by_support:
            failed.append(support)
    return valid_handles, failed


def support_sort_key(support, site_order):
    values = []
    for site_id in support:
        values.append(site_order[site_id])
    return tuple(values)


def next_supports(failed, next_size, site_order):
    failed_set = set(failed)
    joined = set()
    for left, right in combinations(failed, 2):
        site_ids = set(left)
        site_ids.update(right)
        if len(site_ids) != next_size:
            continue

        support = list(site_ids)
        support.sort(key=lambda site_id: site_order[site_id])
        support = tuple(support)

        all_subsets_failed = True
        for subset in combinations(support, next_size - 1):
            if subset not in failed_set:
                all_subsets_failed = False
                break
        if all_subsets_failed:
            joined.add(support)

    supports = list(joined)
    supports.sort(key=lambda support: support_sort_key(support, site_order))
    return supports


def search_handles(ctx, pool, strengths, variable, fit_bank, metric_cal_bank, intervention_bank, r_handle=None, restore_handle=None, self_downstream=False):
    supports = []
    site_order = {}
    for index, row in enumerate(pool):
        site_id = str(row["site_id"])
        supports.append((site_id,))
        site_order[site_id] = index

    all_handles = []
    all_results = []
    all_valid = []
    levels = []
    for size in range(1, ctx.args.max_handle_size + 1):
        if not supports:
            break

        handles = build_handles(ctx, supports, pool, strengths, variable, fit_bank, metric_cal_bank)
        results = evaluate_handles(ctx, intervention_bank, handles, r_handle=r_handle, restore_handle=restore_handle, self_downstream=self_downstream)
        valid_configurations = discovery.get_valid_handles(results, variable, require_restoration=False)
        valid_handles, failed = split_supports(supports, valid_configurations)

        all_handles.extend(handles)
        all_results.extend(results)
        all_valid.extend(valid_handles)

        level = {"k": size, "supports": [], "valid_handles": save_handles(valid_handles), "failed_supports": []}
        for support in supports:
            level["supports"].append(list(support))
        for support in failed:
            level["failed_supports"].append(list(support))
        levels.append(level)

        print(f"  k={size}: supports={len(supports)}, valid={len(valid_handles)}, failed={len(failed)}", flush=True)
        supports = next_supports(failed, size + 1, site_order)

    return all_handles, all_results, all_valid, levels


def node_sort_key(handle):
    return discovery.handle_order(handle)


def name_nodes(handles, variable):
    nodes = []
    for handle in handles:
        nodes.append(dict(handle))
    nodes.sort(key=node_sort_key)

    if len(nodes) == 1:
        nodes[0]["name"] = variable
        return nodes

    nodes[0]["name"] = f"{variable}_early"
    nodes[-1]["name"] = f"{variable}_late"
    middle = list(nodes[1:-1])
    middle.reverse()
    for index, node in enumerate(middle, 1):
        node["name"] = f"{variable}_mid_{index}"
    return nodes


def matching_configurations(handles, source):
    matches = []
    source_sites = tuple(source["site_ids"])
    for handle in handles:
        if tuple(handle["site_ids"]) == source_sites:
            matches.append(handle)
    return matches


def downstream_passes(ctx, result):
    sensitivity_passed = float(result["sensitivity_score"]) >= ctx.args.sensitivity_threshold
    invariance_passed = float(result["invariance_score"]) >= ctx.args.invariance_threshold
    return bool(result["recovery_passed"] and sensitivity_passed and invariance_passed)


def restoration_passes(result):
    return bool(result.get("restoration_passed"))


def make_graph_edge(source, downstream_name, downstream_handle, result, cached_direct, evaluation_mode):
    bypass_to_y = downstream_name != "Y" and not restoration_passes(result)
    return {
        "source": source["name"],
        "downstream": downstream_name,
        "source_handle": result,
        "downstream_handle": downstream_handle,
        "discovery_result": result,
        "cached_direct": cached_direct,
        "evaluation_mode": evaluation_mode,
        "bypass_to_y": bypass_to_y,
    }


def build_downstream_graph(ctx, bank, nodes, configurations, terminal_name, variable, terminal_handle=None, r_handle=None):
    edges = []
    trials = []

    for source in reversed(nodes):
        later_nodes = []
        source_order = discovery.handle_order(source)
        for downstream in nodes:
            if discovery.handle_order(downstream) > source_order:
                later_nodes.append(downstream)
        later_nodes.sort(key=node_sort_key)

        connected = False
        source_configurations = matching_configurations(configurations, source)
        for downstream in later_nodes:
            results = evaluate_handles(ctx, bank, source_configurations, downstream=downstream, r_handle=r_handle)
            passing = []
            for result in results:
                if downstream_passes(ctx, result):
                    passing.append(result)

            trials.append({"source": source["name"], "downstream": downstream["name"], "results": save_handles(results)})
            if not passing:
                continue

            selected = max(passing, key=discovery.handle_selection_key)
            edges.append(make_graph_edge(source, downstream["name"], downstream, selected, False, "downstream"))
            connected = True
            break

        if connected:
            continue

        if variable == "D":
            edges.append(make_graph_edge(source, terminal_name, terminal_handle, source, True, "self_downstream"))
        else:
            edges.append(make_graph_edge(source, terminal_name, None, source, True, "output"))

    return edges, trials


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
            "restoration_passed": restoration_passes(edge["discovery_result"]),
            "bypass_to_Y": edge["bypass_to_y"],
            "source_handle": discovery.save_handle(edge["source_handle"]),
            "discovery_result": discovery.save_handle(edge["discovery_result"]),
        }
        if edge["downstream_handle"] is not None:
            row["downstream_handle"] = discovery.save_handle(edge["downstream_handle"])
        saved.append(row)
    return saved


def evaluate_graph_edge(ctx, bank, edge, variable, r_handle=None):
    source = edge["source_handle"]
    if edge["evaluation_mode"] == "output":
        return evaluate_handles(ctx, bank, [source])[0]
    if edge["evaluation_mode"] == "self_downstream":
        return evaluate_handles(ctx, bank, [source], r_handle=r_handle, restore_handle=r_handle, self_downstream=True)[0]
    return evaluate_handles(ctx, bank, [source], downstream=edge["downstream_handle"], r_handle=r_handle)[0]


def certify_graph(ctx, bank, edges, variable, r_handle=None):
    results = []
    for edge in edges:
        result = evaluate_graph_edge(ctx, bank, edge, variable, r_handle=r_handle)
        heldout_recovery = downstream_passes(ctx, result)
        heldout_bypass = edge["downstream"] != "Y" and not restoration_passes(result)
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

    strengths = []
    for value in args.strength_values.split(","):
        strengths.append(float(value))
    strengths = tuple(strengths)

    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)

    site_lookup = {}
    for site in circuit.sites:
        site_lookup[site.site_id] = site
    ctx = discovery.Context(args, model, circuit.sites, site_lookup, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

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

    print("[1] Discover R nodes", flush=True)
    r_ranking, r_pool = rank_pool(ctx, coarse_fit, ctx.sites)
    r_handles, r_results, r_valid, r_levels = search_handles(ctx, r_pool, strengths, "R", graded_fit, graded_cal, coarse_cal)
    if not r_valid:
        raise RuntimeError("No valid R handle")
    r_nodes = name_nodes(r_valid, "R")
    r_edges, r_edge_trials = build_downstream_graph(ctx, coarse_cal, r_nodes, r_handles, "Y", "R")
    final_r = r_nodes[0]

    print("[2] Discover D nodes", flush=True)
    d_sites = []
    for site in ctx.sites:
        earlier = discovery.layer_order(site.site_id) < discovery.handle_order(final_r)
        outside_r = site.site_id not in final_r["weights"]
        if earlier and outside_r:
            d_sites.append(site)

    d_ranking, d_pool = rank_pool(ctx, graded_fit, tuple(d_sites), downstream=final_r)
    d_handles, d_results, d_valid, d_levels = search_handles(ctx, d_pool, strengths, "D", graded_fit, graded_cal, graded_cal, r_handle=final_r, restore_handle=final_r, self_downstream=True)
    if not d_valid:
        raise RuntimeError("No valid D handle")
    d_nodes = name_nodes(d_valid, "D")
    d_edges, d_edge_trials = build_downstream_graph(ctx, graded_cal, d_nodes, d_handles, final_r["name"], "D", terminal_handle=final_r, r_handle=final_r)

    print("[3] Final Dte graph certification", flush=True)
    coarse_test_pairs = discovery.build_bracket_pairs(coarse_examples, split="Dte", records_per_relation=100)
    graded_test_pairs = discovery.build_graded_pairs(graded_examples, split="Dte", records_per_relation=64, e_definition="active_depth")
    coarse_test = discovery.make_bank(ctx, coarse_examples, coarse_test_pairs, "Dte")
    graded_test = discovery.make_bank(ctx, graded_examples, graded_test_pairs, "Dte")
    r_test_results = certify_graph(ctx, coarse_test, r_edges, "R")
    d_test_results = certify_graph(ctx, graded_test, d_edges, "D", r_handle=final_r)

    passed = True
    for row in r_test_results:
        if not row["graph_structure_passed"]:
            passed = False
    for row in d_test_results:
        if not row["graph_structure_passed"]:
            passed = False

    graph_edges = expanded_edge_labels(d_edges)
    graph_edges.extend(expanded_edge_labels(r_edges))
    final_model = "; ".join(graph_edges)

    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value

    sparse_conversion = []
    for row in sparse_records:
        sparse_conversion.append(row.to_json())

    detailed = {
        "experiment": "automatic_gradual_discovery_v18",
        "config": config,
        "model_info": model_info,
        "sparse_conversion": sparse_conversion,
        "R": {"ranking": r_ranking, "candidate_pool_sites": r_pool, "search_levels": r_levels, "candidate_handles": save_handles(r_handles), "cal_results": save_handles(r_results), "nodes": save_handles(r_nodes), "edges": save_graph_edges(r_edges), "expanded_edges": expanded_edge_labels(r_edges), "edge_trials": r_edge_trials},
        "D": {"ranking": d_ranking, "candidate_pool_sites": d_pool, "search_levels": d_levels, "candidate_handles": save_handles(d_handles), "cal_results": save_handles(d_results), "nodes": save_handles(d_nodes), "edges": save_graph_edges(d_edges), "expanded_edges": expanded_edge_labels(d_edges), "edge_trials": d_edge_trials},
        "Dte": {"R_edges": save_handles(r_test_results), "D_edges": save_handles(d_test_results), "passed": passed},
        "final_model": final_model,
        "passed": passed,
    }

    handles = {}
    for node in d_nodes:
        handles[node["name"]] = discovery.short_handle(node)
    for node in r_nodes:
        handles[node["name"]] = discovery.short_handle(node)

    summary = {"final_model": final_model, "passed": passed, "handles": handles, "R_edges": save_graph_edges(r_edges), "D_edges": save_graph_edges(d_edges), "detailed_output": "automatic_gradual_discovery_v18_detailed.json"}
    detailed_path = args.out_dir / "automatic_gradual_discovery_v18_detailed.json"
    summary_path = args.out_dir / "automatic_gradual_discovery_v18_summary.json"
    discovery.atomic_json(detailed_path, detailed)
    discovery.atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()