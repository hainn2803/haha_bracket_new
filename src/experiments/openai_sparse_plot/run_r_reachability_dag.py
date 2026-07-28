from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery
from . import run_automatic_dag as dag


def build_reachability(ctx, bank, nodes):
    # Patch every source handle and test every later target handle.
    probe_sites = discovery.get_downstream_sites(ctx, *nodes)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    margins, values = discovery.run_handles_and_measure(ctx, bank, nodes, probe_sites)
    tests, edges = [], []

    for source_id, source in enumerate(nodes):
        for target_id, target in enumerate(nodes):
            if not dag.fully_before(source, target):
                continue

            target_values = discovery.combine_downstream_values(values[source_id], probe_site_ids, target["weights"])
            summary = discovery.summarize_r(bank, margins[source_id], target_values, target)
            sensitivity, invariance = discovery.get_block_scores(summary)
            passed = bool(summary["passes"] and sensitivity >= ctx.args.sensitivity_threshold and invariance >= ctx.args.invariance_threshold)
            test = {"source": source_id, "target": target_id, "summary": summary, "sensitivity_score": sensitivity, "invariance_score": invariance, "passed": passed}
            tests.append(test)
            if passed:
                edges.append(test)

        edges.append({
            "source": source_id,
            "target": len(nodes),
            "summary": source["summary"],
            "sensitivity_score": source["sensitivity_score"],
            "invariance_score": source["invariance_score"],
            "passed": True,
        })

    return tests, edges


def main():
    # Recover R handles on Dfit/Dcal and build their reachability DAG.
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

    ranking = discovery.rank_sites(ctx, coarse_fit_bank, ctx.sites)
    candidate_sites = discovery.get_candidate_pool_sites(ranking["ranked_sites"], args.candidate_pool_size, args.mass_fraction)
    candidate_handles = discovery.build_candidate_handles(candidate_sites, strengths, args.max_handle_size)
    candidate_handles = discovery.add_variable_metrics(candidate_handles, graded_fit_bank, graded_cal_bank, args.graded_threshold)
    cal_results = discovery.evaluate_handles(ctx, coarse_cal_bank, candidate_handles)
    valid_handles = discovery.get_valid_handles(cal_results, "R", require_restoration=False)
    nodes = dag.get_minimal_handles(valid_handles)

    pairwise_tests, reachability_edges = build_reachability(ctx, coarse_cal_bank, nodes)
    final_edges, removed_edges = dag.transitive_reduction(reachability_edges, len(nodes))
    final_pairs = {(row["source"], row["target"]) for row in final_edges}

    print("\nR nodes:")
    for node_id, node in enumerate(nodes):
        print(node_id, dag.label(node_id, nodes), "strength=", node["strength"], "order=", discovery.handle_order(node))

    print("\nPassing reachability edges:")
    for edge in reachability_edges:
        print(dag.label(edge["source"], nodes), "->", dag.label(edge["target"], nodes), "sensitivity=", edge["sensitivity_score"], "invariance=", edge["invariance_score"])

    print("\nRemoved transitive edges:")
    for edge in removed_edges:
        print(dag.label(edge["source"], nodes), "->", dag.label(edge["target"], nodes))

    print("\nReachability DAG:")
    for source, target in sorted(final_pairs):
        print(dag.label(source, nodes), "->", dag.label(target, nodes))

    output = {
        "experiment": "r_reachability_dag_v16",
        "split": "Dfit/Dcal",
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "nodes": [discovery.save_handle(row) for row in nodes],
        "pairwise_tests": pairwise_tests,
        "passing_reachability_edges": reachability_edges,
        "removed_transitive_edges": removed_edges,
        "final_edges": dag.save_edges(final_pairs, nodes, set()),
    }
    output_path = Path(args.out_dir) / "r_reachability_dag_v16.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()