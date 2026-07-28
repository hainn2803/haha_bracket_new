from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery
from .bracket_progressive_model_discovery import layer_order


def selection_key(row):
    # Use the current v10 selection rule.
    if hasattr(discovery, "handle_selection_key"):
        return discovery.handle_selection_key(row)
    return discovery.late_handle_key(row)


def recovery_dominates(subset, handle, tolerance=1e-6):
    # Check whether a smaller handle recovers R at least as well.
    return (
        subset["summary"]["score"] >= handle["summary"]["score"] - tolerance
        and subset["sensitivity_score"] >= handle["sensitivity_score"] - tolerance
        and subset["invariance_score"] >= handle["invariance_score"] - tolerance
    )


def get_minimal_handles(handles):
    # Keep one strength per handle and remove recovery-redundant supersets.
    best = {}
    for row in handles:
        sites = tuple(sorted(row["site_ids"]))
        if sites not in best or selection_key(row) > selection_key(best[sites]):
            best[sites] = row

    unique = list(best.values())
    minimal = []
    for row in unique:
        sites = set(row["site_ids"])
        subsets = [other for other in unique if set(other["site_ids"]) < sites]
        if not any(recovery_dominates(other, row) for other in subsets):
            minimal.append(row)
    return sorted(minimal, key=discovery.handle_order)


def fully_before(source, target):
    # Require every source site to occur before every target site.
    source_order = max(layer_order(site_id) for site_id in source["site_ids"])
    target_order = min(layer_order(site_id) for site_id in target["site_ids"])
    return source_order < target_order


def build_reachability(ctx, bank, nodes):
    # Intervene on every node and probe all node sites in one batched evaluation.
    probe_sites = discovery.get_downstream_sites(ctx, *nodes)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    margins, values = discovery.run_handles_and_measure(ctx, bank, nodes, probe_sites)
    edges = []

    for source_id, source in enumerate(nodes):
        for target_id, target in enumerate(nodes):
            if not fully_before(source, target):
                continue

            downstream_values = discovery.combine_downstream_values(values[source_id], probe_site_ids, target["weights"])
            summary = discovery.summarize_r(bank, margins[source_id], downstream_values, target)
            sensitivity, invariance = discovery.get_block_scores(summary)
            passed = bool(summary["passes"] and sensitivity >= ctx.args.sensitivity_threshold and invariance >= ctx.args.invariance_threshold)

            if passed:
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "summary": summary,
                    "sensitivity_score": sensitivity,
                    "invariance_score": invariance,
                })

        edges.append({
            "source": source_id,
            "target": len(nodes),
            "summary": source["summary"],
            "sensitivity_score": source["sensitivity_score"],
            "invariance_score": source["invariance_score"],
        })

    return edges, margins, probe_sites


def transitive_reduction(edges, node_count):
    # Remove A -> C when the graph already contains A -> B -> C.
    pairs = {(row["source"], row["target"]) for row in edges}
    reduced, removed = [], []

    for row in edges:
        source, target = row["source"], row["target"]
        indirect = any((source, middle) in pairs and (middle, target) in pairs for middle in range(node_count) if middle not in (source, target))
        (removed if indirect else reduced).append(row)

    return reduced, removed


def restore_children(ctx, bank, source, direct_margins, probe_sites, children):
    # Restore a set of downstream children and measure the remaining output effect.
    restore_sites = tuple(sorted({site_id for child in children for site_id in child["site_ids"]}))
    restored_margins, _ = discovery.run_handles_and_measure(ctx, bank, [source], probe_sites, restore_sites)
    return discovery.mediation_summary(bank.pairs, bank.by_id, bank.runs, direct_margins, restored_margins[0])


def apply_group_restoration(ctx, bank, nodes, reduced_edges, direct_margins, probe_sites):
    # Remove redundant children and add A -> Y when known children miss a bypass.
    y_id = len(nodes)
    pairs = {(row["source"], row["target"]) for row in reduced_edges}
    records = []

    for source_id, source in enumerate(nodes):
        child_ids = sorted(target for start, target in pairs if start == source_id and target != y_id)
        if not child_ids:
            continue

        children = [nodes[target_id] for target_id in child_ids]
        group_summary = restore_children(ctx, bank, source, direct_margins[source_id], probe_sites, children)
        removed_children = []

        if group_summary["passes"]:
            for child_id in list(child_ids):
                remaining_ids = [target_id for target_id in child_ids if target_id != child_id]
                remaining = [nodes[target_id] for target_id in remaining_ids]
                trial_summary = restore_children(ctx, bank, source, direct_margins[source_id], probe_sites, remaining)
                if trial_summary["passes"]:
                    pairs.remove((source_id, child_id))
                    child_ids = remaining_ids
                    removed_children.append(child_id)
            group_summary = restore_children(ctx, bank, source, direct_margins[source_id], probe_sites, [nodes[target_id] for target_id in child_ids])
        else:
            pairs.add((source_id, y_id))

        records.append({
            "source": source_id,
            "children": child_ids,
            "removed_redundant_children": removed_children,
            "restoration": group_summary,
            "bypass_to_Y": not group_summary["passes"],
        })

    return pairs, records


def label(node_id, nodes):
    # Print Y or the sites of one recovered R node.
    if node_id == len(nodes):
        return "Y"
    return "{" + " + ".join(nodes[node_id]["site_ids"]) + "}"


def save_edges(pairs, nodes, bypass_sources):
    # Convert internal node indices into readable edges.
    return [{
        "source_id": source,
        "target_id": target,
        "source": label(source, nodes),
        "target": label(target, nodes),
        "bypass": source in bypass_sources and target == len(nodes),
    } for source, target in sorted(pairs)]


def main():
    # Discover recovered R nodes on Dfit/Dcal and organize them into a DAG.
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
    nodes = get_minimal_handles(valid_handles)

    reachability_edges, direct_margins, probe_sites = build_reachability(ctx, coarse_cal_bank, nodes)
    reduced_edges, removed_transitive_edges = transitive_reduction(reachability_edges, len(nodes))
    final_pairs, restoration_records = apply_group_restoration(ctx, coarse_cal_bank, nodes, reduced_edges, direct_margins, probe_sites)
    bypass_sources = {row["source"] for row in restoration_records if row["bypass_to_Y"]}
    final_edges = save_edges(final_pairs, nodes, bypass_sources)

    print("\nR DAG nodes:")
    for node_id, node in enumerate(nodes):
        print(node_id, label(node_id, nodes), "strength=", node["strength"], "order=", discovery.handle_order(node))

    print("\nR DAG edges:")
    for edge in final_edges:
        suffix = " [bypass]" if edge["bypass"] else ""
        print(edge["source"], "->", edge["target"] + suffix)

    print("\nGroup restoration:")
    for row in restoration_records:
        print({
            "source": label(row["source"], nodes),
            "children": [label(child_id, nodes) for child_id in row["children"]],
            "restored_base": row["restoration"]["restored_Rmid_output_preserves_base"],
            "removed_fraction": row["restoration"]["mean_output_effect_removed_fraction"],
            "passes": row["restoration"]["passes"],
            "bypass_to_Y": row["bypass_to_Y"],
        })

    output = {
        "experiment": "r_dag_discovery_v13",
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "nodes": [discovery.save_handle(row) for row in nodes],
        "reachability_edges": reachability_edges,
        "removed_transitive_edges": removed_transitive_edges,
        "group_restoration": restoration_records,
        "final_edges": final_edges,
    }
    output_path = Path(args.out_dir) / "r_dag_discovery_v13.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()