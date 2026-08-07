import math
from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


THRESHOLD = 0.90

PATCH_HANDLE = {
    "name": "R_mid2",
    "handle_id": "patch_R_mid2",
    "site_ids": ["2.attn.resid_delta:1249"],
    "weights": {"2.attn.resid_delta:1249": 2.0},
    "strength": 2.0,
}

RESTORE_HANDLES = [
    # {
    #     "name": "R_mid1",
    #     "site_ids": ["final_resid:1079"],
    #     "weights": {"final_resid:1079": 1.0},
    #     "strength": 1.0,
    # },
    # {
    #     "name": "R_late",
    #     "site_ids": ["final_resid:2041"],
    #     "weights": {"final_resid:2041": 1.0},
    #     "strength": 1.0,
    # },
    {
        "name": "R_late",
        "site_ids": ["4.attn.act_in:1249"],
        "weights": {"4.attn.act_in:1249": 1.0},
        "strength": 1.0,
    },
]


def wilson_interval(successes, total):
    # Compute a 95% confidence interval for a rate.
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / scale
    return [center - radius, center + radius]


def rate_summary(matches):
    # Return a rate and its 95% confidence interval.
    successes, total = sum(matches), len(matches)
    return {"successes": successes, "total": total, "rate": successes / total if total else 0.0, "ci95": wilson_interval(successes, total)}


def output_rates(bank, direct_margins, restored_margins):
    # Measure the direct and remaining source effects on different-R records.
    direct_source, restored_source, restored_base = [], [], []

    for i, pair in enumerate(bank.pairs):
        base = bank.by_id[pair.base_id].variable_value
        source = bank.by_id[pair.source_id].variable_value
        if source == base:
            continue

        direct_output = 1 if float(direct_margins[i]) > 0 else -1
        restored_output = 1 if float(restored_margins[i]) > 0 else -1
        direct_source.append(direct_output == source)
        restored_source.append(restored_output == source)
        restored_base.append(restored_output == base)

    return {
        "direct_output_matches_source": rate_summary(direct_source),
        "restored_output_matches_source": rate_summary(restored_source),
        "restored_output_matches_base": rate_summary(restored_base),
    }


def main():
    # Patch one handle and restore a list of handles on heldout Dte.
    args = discovery.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, _ = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    discovery.convert_transformer_linears_to_sparse(model)
    ctx = discovery.Context(args, model, circuit.sites, {site.site_id: site for site in circuit.sites}, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    examples = discovery.build_bracket_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    test_bank = discovery.make_bank(ctx, examples, discovery.build_bracket_pairs(examples, split="Dte", records_per_relation=100), "Dte")

    restore_site_ids = tuple(dict.fromkeys(site_id for handle in RESTORE_HANDLES for site_id in handle["site_ids"]))
    probe_sites = tuple(ctx.site_lookup[site_id] for site_id in restore_site_ids)
    direct_margins, _ = discovery.run_handles_and_measure(ctx, test_bank, [PATCH_HANDLE], probe_sites)
    restored_margins, _ = discovery.run_handles_and_measure(ctx, test_bank, [PATCH_HANDLE], probe_sites, restore_site_ids)

    direct_margins, restored_margins = direct_margins[0], restored_margins[0]
    rates = output_rates(test_bank, direct_margins, restored_margins)
    restoration = discovery.mediation_summary(test_bank.pairs, test_bank.by_id, test_bank.runs, direct_margins, restored_margins)
    direct_passed = rates["direct_output_matches_source"]["rate"] >= THRESHOLD
    bypass_passed = direct_passed and rates["restored_output_matches_source"]["rate"] >= THRESHOLD

    result = {
        "experiment": "bypass_restoration_test_v14",
        "split": "Dte",
        "threshold": THRESHOLD,
        "patch_handle": PATCH_HANDLE,
        "restore_handles": RESTORE_HANDLES,
        "restore_site_ids": restore_site_ids,
        "rates": rates,
        "mean_output_effect_removed_fraction": restoration["mean_output_effect_removed_fraction"],
        "mean_output_effect_remaining_fraction": 1.0 - restoration["mean_output_effect_removed_fraction"],
        "complete_mediation_passed": restoration["passes"],
        "bypass_passed": bypass_passed,
    }

    output_path = Path(args.out_dir) / "bypass_restoration_test_v14.json"
    discovery.atomic_json(output_path, result)
    print("\nPatch:", PATCH_HANDLE["name"], PATCH_HANDLE["site_ids"])
    print("Restore:", [(handle["name"], handle["site_ids"]) for handle in RESTORE_HANDLES])
    print("Direct source match:", rates["direct_output_matches_source"])
    print("Restored source match:", rates["restored_output_matches_source"])
    print("Restored base match:", rates["restored_output_matches_base"])
    print("Removed effect:", result["mean_output_effect_removed_fraction"])
    print("Remaining effect:", result["mean_output_effect_remaining_fraction"])
    print("Complete mediation passed:", result["complete_mediation_passed"])
    print("Bypass passed:", result["bypass_passed"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()