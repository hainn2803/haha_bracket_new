from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


def main():
    args = discovery.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = tuple(float(x) for x in args.strength_values.split(","))

    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, _ = discovery.load_sparse_gpt_model(
        model_name="csp_yolo2",
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=True,
        grad_checkpointing=False,
    )
    discovery.convert_transformer_linears_to_sparse(model)
    ctx = discovery.Context(
        args,
        model,
        circuit.sites,
        {site.site_id: site for site in circuit.sites},
        int(encoder.encode("]\n")[0]),
        int(encoder.encode("]]\n")[0]),
        "cuda" if args.cuda else "cpu",
    )

    coarse = discovery.build_bracket_rediscovery_bank(
        encoder,
        fit_contents=48,
        cal_contents=24,
        test_contents=24,
        content_offset=13000,
    )
    cal_bank = discovery.make_bank(
        ctx,
        coarse,
        discovery.build_bracket_pairs(
            coarse, split="Dcal", records_per_relation=100,
        ),
        "Dcal",
    )

    graded = discovery.build_graded_d_bank(
        encoder,
        fit_contents=16,
        cal_contents=8,
        test_contents=8,
        content_offset=17000,
        q_grid=(0, 1, 2, 4),
    )
    fit_bank = discovery.make_bank(
        ctx,
        graded,
        discovery.build_graded_pairs(
            graded,
            split="Dfit",
            records_per_relation=64,
            e_definition="active_depth",
        ),
        "Dfit",
    )
    graded_cal_bank = discovery.make_bank(
        ctx,
        graded,
        discovery.build_graded_pairs(
            graded,
            split="Dcal",
            records_per_relation=48,
            e_definition="active_depth",
        ),
        "Dcal",
    )

    handles = [
        {
            "handle_id": f"k1_{i}",
            "site_ids": [site.site_id],
            "weights": {site.site_id: 1.0},
            "k": 1,
            "strength": 1.0,
            "ot_mass": None,
        }
        for i, site in enumerate(ctx.sites, 1)
    ]
    handles = discovery.add_variable_metrics(
        handles, fit_bank, graded_cal_bank, args.graded_threshold,
    )
    handles = [
        {
            **handle,
            "handle_id": f"{handle['handle_id']}_s{strength:g}",
            "strength": strength,
        }
        for handle in handles
        for strength in strengths
    ]
    results = discovery.evaluate_handles(ctx, cal_bank, handles)

    best = {}
    for row in results:
        site_id = row["site_ids"][0]
        if site_id not in best or discovery.handle_selection_key(row) > discovery.handle_selection_key(best[site_id]):
            best[site_id] = row

    valid_sites = {
        row["site_ids"][0]
        for row in discovery.get_valid_handles(results, "R")
    }
    sites = [
        {
            **discovery.save_handle(row),
            "valid_R": site_id in valid_sites,
        }
        for site_id, row in best.items()
    ]
    sites.sort(
        key=lambda row: (
            row["valid_R"],
            min(row["r_accuracy"].values()),
            row["summary"]["score"],
        ),
        reverse=True,
    )

    for i, row in enumerate(sites, 1):
        print(i, {
            "site": row["site_ids"][0],
            "strength": row["strength"],
            "valid_R": row["valid_R"],
            "R_accuracy": row["r_accuracy"],
            "is_D": row["is_D"],
            "recovery": row["summary"]["score"],
            "sensitivity": row["sensitivity_score"],
            "invariance": row["invariance_score"],
        })

    output = {
        "experiment": "all_133_r_alignment_v2",
        "sites": sites,
        "results_by_strength": [
            discovery.save_handle(row)
            for row in results
        ],
    }
    output_path = Path(args.out_dir) / "all_133_r_alignment_v2.json"
    discovery.atomic_json(output_path, output)
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()