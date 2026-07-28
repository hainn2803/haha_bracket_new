from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery
from . import run_automatic_dag as dag
from . import run_r_reachability_dag as r_dag
from .bracket_progressive_model_discovery import layer_order


def label(node_id, nodes):
    # Print X, Y, or one D/R handle.
    if node_id == "X":
        return "X"
    if node_id == len(nodes):
        return "Y"
    node = nodes[node_id]
    return node["variable"] + " {" + " + ".join(node["site_ids"]) + "}"


def d_scores(summary):
    # Split graded-D validation into sensitivity and invariance scores.
    blocks = summary["balanced_blocks"]
    sensitivity = min(blocks["E_correct"], blocks["cross_Rmid"], blocks["cross_output"])
    invariance = min(blocks["E_correct"], blocks["same_side_Rmid"], blocks["same_side_output"])
    return sensitivity, invariance


def build_d_reachability(ctx, bank, d_nodes, r_nodes, r_root):
    # Patch every D node and test all later D and R nodes.
    nodes = d_nodes + r_nodes
    probe_sites = discovery.get_downstream_sites(ctx, *nodes)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    margins, values = discovery.run_handles_and_measure(ctx, bank, d_nodes, probe_sites)
    tests, edges = [], []

    for source_id, source in enumerate(d_nodes):
        for target_id, target in enumerate(nodes):
            if not dag.fully_before(source, target):
                continue

            if target["variable"] == "D":
                summary = discovery.summarize_d(bank, margins[source_id], values[source_id], probe_site_ids, target, r_root)
                sensitivity, invariance = d_scores(summary)
            else:
                target_values = discovery.combine_downstream_values(values[source_id], probe_site_ids, target["weights"])
                summary = discovery.summarize_r(bank, margins[source_id], target_values, target)
                sensitivity, invariance = discovery.get_block_scores(summary)

            passed = bool(summary["passes"] and sensitivity >= ctx.args.sensitivity_threshold and invariance >= ctx.args.invariance_threshold)
            test = {"source": source_id, "target": target_id, "edge_type": f"D->{target['variable']}", "summary": summary, "sensitivity_score": sensitivity, "invariance_score": invariance, "passed": passed}
            tests.append(test)
            if passed:
                edges.append(test)

        edges.append({
            "source": source_id,
            "target": len(nodes),
            "edge_type": "D->Y",
            "summary": source["summary"],
            "sensitivity_score": source["sensitivity_score"],
            "invariance_score": source["invariance_score"],
            "passed": True,
        })

    return tests, edges


def move_r_ids(rows, d_count, r_count):
    # Move local R node IDs after the D node IDs.
    output = []
    for row in rows:
        target = d_count + r_count if row["target"] == r_count else d_count + row["target"]
        output.append({**row, "source": d_count + row["source"], "target": target, "edge_type": "R->Y" if target == d_count + r_count else "R->R"})
    return output


def save_edges(pairs, nodes):
    # Convert internal node IDs into readable full-DAG edges.
    return [{"source_id": source, "target_id": target, "source": label(source, nodes), "target": label(target, nodes)} for source, target in sorted(pairs)]


def main():
    # Recover D and R handles on Dfit/Dcal and build the full reachability DAG.
    args = discovery.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = tuple(float(value) for value in args.strength_values.split(","))
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)
    ctx = discovery.Context(args, model, circuit.sites, {site.site_id: site for site in circuit.sites}, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    coarse_examples = discovery.build_bracket_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse_fit_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dfit", records_per_relation=100), "Dfit")
    coarse_cal_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dcal", records_per_relation=100), "Dcal")
    graded_examples = discovery.build_graded_d_bank(encoder, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    graded_fit_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dfit", records_per_relation=64, e_definition="active_depth"), "Dfit")
    graded_cal_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dcal", records_per_relation=48, e_definition="active_depth"), "Dcal")

    r_ranking = discovery.rank_sites(ctx, coarse_fit_bank, ctx.sites)
    r_sites = discovery.get_candidate_pool_sites(r_ranking["ranked_sites"], args.candidate_pool_size, args.mass_fraction)
    r_handles = discovery.build_candidate_handles(r_sites, strengths, args.max_handle_size)
    r_handles = discovery.add_variable_metrics(r_handles, graded_fit_bank, graded_cal_bank, args.graded_threshold)
    r_results = discovery.evaluate_handles(ctx, coarse_cal_bank, r_handles)
    r_nodes = [{**row, "variable": "R"} for row in dag.get_minimal_handles(discovery.get_valid_handles(r_results, "R", require_restoration=False))]

    r_tests, r_edges = r_dag.build_reachability(ctx, coarse_cal_bank, r_nodes)
    r_targets = {row["target"] for row in r_edges if row["target"] < len(r_nodes)}
    r_root_id = min((i for i in range(len(r_nodes)) if i not in r_targets), key=lambda i: discovery.handle_order(r_nodes[i]))
    r_root = r_nodes[r_root_id]

    d_search_sites = tuple(site for site in ctx.sites if layer_order(site.site_id) < discovery.handle_order(r_root) and site.site_id not in r_root["weights"])
    d_ranking = discovery.rank_sites(ctx, graded_fit_bank, d_search_sites, downstream_handle=r_root)
    d_sites = discovery.get_candidate_pool_sites(d_ranking["ranked_sites"], args.candidate_pool_size, args.mass_fraction)
    d_handles = discovery.build_candidate_handles(d_sites, strengths, args.max_handle_size)
    d_handles = discovery.add_variable_metrics(d_handles, graded_fit_bank, graded_cal_bank, args.graded_threshold)
    d_handles = [{**row, "variable": "D"} for row in d_handles if row["is_D"]]
    d_results = [discovery.evaluate_handles(ctx, graded_cal_bank, [row], downstream_handle=row, r_handle=r_root, restore_handle=r_root)[0] for row in d_handles]
    d_nodes = [{**row, "variable": "D"} for row in dag.get_minimal_handles(discovery.get_valid_handles(d_results, "D", require_restoration=False))]

    nodes = d_nodes + r_nodes
    d_tests, d_edges = build_d_reachability(ctx, graded_cal_bank, d_nodes, r_nodes, r_root)
    r_tests = move_r_ids(r_tests, len(d_nodes), len(r_nodes))
    r_edges = move_r_ids(r_edges, len(d_nodes), len(r_nodes))
    all_tests = d_tests + r_tests
    all_edges = d_edges + r_edges
    reduced_edges, removed_edges = dag.transitive_reduction(all_edges, len(nodes))
    final_pairs = {(row["source"], row["target"]) for row in reduced_edges}
    incoming = {target for _, target in final_pairs if target < len(nodes)}
    root_ids = [i for i in range(len(nodes)) if i not in incoming]
    boundary_edges = [{"source": "X", "target": label(i, nodes), "target_id": i} for i in root_ids]

    print("\nFull DAG nodes:")
    for node_id, node in enumerate(nodes):
        print(node_id, label(node_id, nodes), "strength=", node["strength"], "order=", discovery.handle_order(node))
    print("R root used for D discovery:", label(len(d_nodes) + r_root_id, nodes))

    print("\nPassing reachability edges:")
    for edge in all_edges:
        print(label(edge["source"], nodes), "->", label(edge["target"], nodes), "sensitivity=", edge["sensitivity_score"], "invariance=", edge["invariance_score"])

    print("\nRemoved transitive edges:")
    for edge in removed_edges:
        print(label(edge["source"], nodes), "->", label(edge["target"], nodes))

    print("\nFull reachability DAG:")
    for edge in boundary_edges:
        print("X ->", edge["target"])
    for source, target in sorted(final_pairs):
        print(label(source, nodes), "->", label(target, nodes))

    output = {
        "experiment": "full_reachability_dag",
        "split": "Dfit/Dcal",
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "R_root_for_D_discovery": discovery.save_handle(r_root),
        "D_nodes": [discovery.save_handle(row) for row in d_nodes],
        "R_nodes": [discovery.save_handle(row) for row in r_nodes],
        "pairwise_tests": all_tests,
        "passing_reachability_edges": all_edges,
        "removed_transitive_edges": removed_edges,
        "boundary_edges": boundary_edges,
        "final_edges": save_edges(final_pairs, nodes),
    }
    output_path = Path(args.out_dir) / "full_reachability_dag.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()