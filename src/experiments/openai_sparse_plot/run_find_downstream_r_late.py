from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery
from .bracket_progressive_model_discovery import layer_order


SOURCE_SITE_ID = "7.mlp.act_in:1079"


def selection_key(row):
    # Use the current handle selection rule from v10.
    if hasattr(discovery, "handle_selection_key"):
        return discovery.handle_selection_key(row)
    return discovery.late_handle_key(row)


def get_perfect_handles(handles):
    # Keep one best strength for each perfect R handle.
    best = {}
    for row in handles:
        if row["summary"]["score"] != 1.0 or row["sensitivity_score"] != 1.0 or row["invariance_score"] != 1.0:
            continue
        sites = tuple(sorted(row["site_ids"]))
        if sites not in best or selection_key(row) > selection_key(best[sites]):
            best[sites] = row
    return sorted(best.values(), key=selection_key, reverse=True)


def get_source_handle(handles):
    # Select the best singleton configuration for 7.mlp.act_in:1079.
    candidates = [row for row in handles if row["site_ids"] == [SOURCE_SITE_ID]]
    assert candidates, f"No valid singleton handle found for {SOURCE_SITE_ID}"
    return max(candidates, key=selection_key)


def is_fully_downstream(handle, source_handle):
    # Require every target site to occur after the fixed source handle.
    source_order = discovery.handle_order(source_handle)
    return all(layer_order(site_id) > source_order for site_id in handle["site_ids"])


def restoration_score(row):
    # Read the fraction of the source output effect removed by restoring the target.
    return float(row["restoration"]["mean_output_effect_removed_fraction"])


def test_downstream_handles(ctx, cal_bank, valid_handles):
    # Test 7.mlp.act_in:1079 -> each later perfect R handle -> Y.
    source = get_source_handle(valid_handles)
    targets = [row for row in get_perfect_handles(valid_handles) if is_fully_downstream(row, source) and SOURCE_SITE_ID not in row["site_ids"]]
    trials = []

    for target in targets:
        downstream = {**target, "variable": "R"}
        result = discovery.evaluate_handles(ctx, cal_bank, [source], downstream_handle=downstream)[0]
        trials.append({"downstream_handle": discovery.save_handle(downstream), "edge_result": discovery.save_handle(result)})

    trials.sort(key=lambda row: (
        row["edge_result"]["edge_certified"],
        row["edge_result"]["recovery_passed"],
        row["edge_result"]["restoration_passed"],
        row["edge_result"]["sensitivity_score"],
        row["edge_result"]["invariance_score"],
        restoration_score(row["edge_result"]),
    ), reverse=True)

    print("\nSource R handle:", source["site_ids"], "strength=", source["strength"])
    for i, trial in enumerate(trials, 1):
        target, result = trial["downstream_handle"], trial["edge_result"]
        print(i, {
            "edge": f"{SOURCE_SITE_ID} -> {target['site_ids']}",
            "target_strength": target["strength"],
            "sensitivity": result["sensitivity_score"],
            "invariance": result["invariance_score"],
            "restored_base": result["restoration"]["restored_Rmid_output_preserves_base"],
            "removed_fraction": result["restoration"]["mean_output_effect_removed_fraction"],
            "edge_certified": result["edge_certified"],
        })

    return source, trials


def main():
    # Build R on Dfit/Dcal and test its possible downstream handles.
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
    source, trials = test_downstream_handles(ctx, coarse_cal_bank, valid_handles)

    output = {
        "experiment": "run_find_downstream_r_late",
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "source_handle": discovery.save_handle(source),
        "trials": trials,
    }
    output_path = Path(args.out_dir) / "run_find_downstream_r_late.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()