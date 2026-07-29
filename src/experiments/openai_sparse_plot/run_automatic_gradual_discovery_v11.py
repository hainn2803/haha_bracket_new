from __future__ import annotations

import json
from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


def build_candidates(ctx, fit_bank, graded_fit_bank, graded_cal_bank, sites, strengths, variable, downstream=None):
    # Rank eligible sites and build new handles for this round.
    ranking = discovery.rank_sites(ctx, fit_bank, sites, downstream_handle=downstream)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)
    handles = discovery.build_candidate_handles(pool, strengths, ctx.args.max_handle_size)
    handles = discovery.add_variable_metrics(handles, graded_fit_bank, graded_cal_bank, ctx.args.graded_threshold)
    if variable == "D":
        handles = [{**row, "variable": "D"} for row in handles if row["is_D"]]
    return ranking, pool, handles


def refine_handles(ctx, fit_bank, cal_bank, graded_fit_bank, graded_cal_bank, sites, late_handle, strengths, r_handle=None, require_restoration=False):
    # Freeze one handle, rebuild candidates before it, and repeat.
    variable = late_handle["variable"]
    chain, rounds, current = [late_handle], [], late_handle
    used_sites = set(current["site_ids"])

    while True:
        eligible = tuple(site for site in sites if discovery.layer_order(site.site_id) < discovery.handle_order(current) and site.site_id not in used_sites)
        if not eligible:
            break

        ranking, pool, candidates = build_candidates(ctx, fit_bank, graded_fit_bank, graded_cal_bank, eligible, strengths, variable, downstream=current)
        results = discovery.evaluate_handles(ctx, cal_bank, candidates, downstream_handle=current, r_handle=r_handle) if candidates else []
        valid = discovery.get_valid_handles(results, variable, require_restoration=require_restoration)
        rounds.append({
            "downstream_handle": discovery.save_handle(current),
            "eligible_site_ids": [site.site_id for site in eligible],
            "ranking": ranking,
            "candidate_pool_sites": pool,
            "candidate_handles": [discovery.save_handle(row) for row in candidates],
            "results": [discovery.save_handle(row) for row in results],
            "valid_handles": [discovery.save_handle(row) for row in valid],
        })

        if not valid:
            break

        current = {**max(valid, key=discovery.handle_selection_key), "variable": variable}
        chain.append(current)
        used_sites.update(current["site_ids"])

    return chain, rounds


def main():
    # Discover R and D, rebuilding candidates after every frozen handle.
    args = discovery.parse_args()
    if args.out_dir == Path("outputs/automatic_gradual_discovery_v10"):
        args.out_dir = Path("outputs/automatic_gradual_discovery_v12")
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

    # Discover R.
    print("[1] Discover R", flush=True)
    r_ranking, r_pool, r_handles = build_candidates(ctx, coarse_fit_bank, graded_fit_bank, graded_cal_bank, ctx.sites, strengths, "R")
    r_cal_results = discovery.evaluate_handles(ctx, coarse_cal_bank, r_handles)
    r_valid_handles = discovery.get_valid_handles(r_cal_results, "R")
    discovery.print_discovery_results("R", r_handles, r_cal_results, r_valid_handles, discovery.handle_order, discovery.handle_selection_key)
    assert r_valid_handles, "No R handle passed Dcal recovery"

    r_late = discovery.select_late_handle(r_valid_handles, "R")
    r_chain, r_refinement = refine_handles(ctx, coarse_fit_bank, coarse_cal_bank, graded_fit_bank, graded_cal_bank, ctx.sites, r_late, strengths)
    discovery.name_chain(r_chain, "R")
    final_r = r_chain[-1]
    discovery.print_refinement("R", r_chain, r_refinement, discovery.handle_order)
    discovery.print_frozen_handle("R", final_r, discovery.handle_order)

    # Discover D.
    print("[2] Discover D", flush=True)
    d_sites = tuple(site for site in ctx.sites if discovery.layer_order(site.site_id) < discovery.handle_order(final_r) and site.site_id not in final_r["weights"])
    d_ranking, d_pool, d_handles = build_candidates(ctx, graded_fit_bank, graded_fit_bank, graded_cal_bank, d_sites, strengths, "D", downstream=final_r)
    d_cal_results = [discovery.evaluate_handles(ctx, graded_cal_bank, [handle], downstream_handle=handle, r_handle=final_r, restore_handle=final_r)[0] for handle in d_handles]
    d_valid_handles = discovery.get_valid_handles(d_cal_results, "D", require_restoration=True)
    discovery.print_discovery_results("D", d_handles, d_cal_results, d_valid_handles, discovery.handle_order, discovery.handle_selection_key)
    assert d_valid_handles, "No D handle passed Dcal recovery"

    d_late = discovery.select_late_handle(d_valid_handles, "D")
    d_chain, d_refinement = refine_handles(ctx, graded_fit_bank, graded_cal_bank, graded_fit_bank, graded_cal_bank, ctx.sites, d_late, strengths, r_handle=final_r)
    discovery.name_chain(d_chain, "D")
    final_d = d_chain[-1]
    discovery.print_refinement("D", d_chain, d_refinement, discovery.handle_order)
    discovery.print_frozen_handle("D", final_d, discovery.handle_order)

    # Certify the frozen chains once on Dte.
    print("[3] Final Dte certification", flush=True)
    coarse_test_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dte", records_per_relation=100), "Dte")
    graded_test_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dte", records_per_relation=64, e_definition="active_depth"), "Dte")
    r_test_results = discovery.certify_chain(ctx, coarse_test_bank, r_chain)
    d_test_results = discovery.certify_chain(ctx, graded_test_bank, d_chain, r_handle=final_r)
    passed = all(row["edge_certified"] for row in r_test_results + d_test_results)

    final_model = "X -> " + " -> ".join([row["name"] for row in reversed(d_chain)] + [row["name"] for row in reversed(r_chain)]) + " -> Y"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    detailed = {
        "experiment": "automatic_gradual_discovery_v12", "config": config, "model_info": model_info, "sparse_conversion": [row.to_json() for row in sparse_records],
        "R": {"ranking": r_ranking, "candidate_pool_sites": r_pool, "candidate_handles": [discovery.save_handle(row) for row in r_handles], "cal_results": [discovery.save_handle(row) for row in r_cal_results], "valid_handles": [discovery.save_handle(row) for row in r_valid_handles], "refinement": r_refinement, "chain": [discovery.save_handle(row) for row in r_chain]},
        "D": {"ranking": d_ranking, "candidate_pool_sites": d_pool, "candidate_handles": [discovery.save_handle(row) for row in d_handles], "cal_results": [discovery.save_handle(row) for row in d_cal_results], "valid_handles": [discovery.save_handle(row) for row in d_valid_handles], "refinement": d_refinement, "chain": [discovery.save_handle(row) for row in d_chain]},
        "Dte": {"R_edges": [discovery.save_handle(row) for row in r_test_results], "D_edges": [discovery.save_handle(row) for row in d_test_results], "passed": passed},
        "banks": {"coarse": discovery.bank_manifest(coarse_examples, {"Dfit": coarse_fit_bank.pairs, "Dcal": coarse_cal_bank.pairs, "Dte": coarse_test_bank.pairs}), "graded": discovery.bank_manifest(graded_examples, {"Dfit": graded_fit_bank.pairs, "Dcal": graded_cal_bank.pairs, "Dte": graded_test_bank.pairs})},
        "clean_accuracy": {"coarse": {bank.name: discovery.clean_accuracy(bank.examples, bank.runs, split=bank.name) for bank in (coarse_fit_bank, coarse_cal_bank, coarse_test_bank)}, "graded": {bank.name: discovery.clean_accuracy(bank.examples, bank.runs, split=bank.name) for bank in (graded_fit_bank, graded_cal_bank, graded_test_bank)}},
        "final_model": final_model, "passed": passed,
    }
    summary = {"final_model": final_model, "passed": passed, "handles": {row["name"]: discovery.short_handle(row) for row in d_chain + r_chain}, "Dte_edges": {row["edge"]: row["edge_certified"] for row in r_test_results + d_test_results}, "detailed_output": "automatic_gradual_discovery_v12_detailed.json"}
    detailed_path = args.out_dir / "automatic_gradual_discovery_v12_detailed.json"
    summary_path = args.out_dir / "automatic_gradual_discovery_v12_summary.json"
    discovery.atomic_json(detailed_path, detailed)
    discovery.atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()