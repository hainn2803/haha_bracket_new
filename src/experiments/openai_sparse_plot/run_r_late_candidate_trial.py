from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


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


def test_r_late_handles(ctx, cal_bank, valid_handles):
    # Try every perfect R handle as R_late and require certified refinement edges.
    perfect_handles = get_perfect_handles(valid_handles)
    trials = []

    for i, handle in enumerate(perfect_handles, 1):
        r_late = {**handle, "variable": "R"}
        chain, rounds = discovery.refine_handles(ctx, cal_bank, valid_handles, r_late, require_restoration=True)
        discovery.name_chain(chain, "R")

        print(f"\n[{i}/{len(perfect_handles)}] R_late {r_late['site_ids']}, strength={r_late['strength']}")
        for row in chain:
            print(row["name"], row["site_ids"], row["strength"], discovery.handle_order(row))

        trials.append({
            "R_late": discovery.save_handle(r_late),
            "chain": [discovery.save_handle(row) for row in chain],
            "refinement": rounds,
        })

    return trials


def main():
    # Build Dfit/Dcal and compare all perfect R_late candidates.
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
    perfect_handles = get_perfect_handles(valid_handles)

    print("Valid R handles:", len(valid_handles))
    print("Perfect distinct R handles:", len(perfect_handles))
    trials = test_r_late_handles(ctx, coarse_cal_bank, valid_handles)

    output = {
        "experiment": "r_late_candidate_trials_v11",
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "candidate_pool_sites": candidate_sites,
        "perfect_handles": [discovery.save_handle(row) for row in perfect_handles],
        "trials": trials,
    }
    output_path = Path(args.out_dir) / "r_late_candidate_trials_v11.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()