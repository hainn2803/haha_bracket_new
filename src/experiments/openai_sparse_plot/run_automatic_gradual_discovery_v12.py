from __future__ import annotations

import json
from pathlib import Path

from . import run_automatic_gradual_discovery_v10 as discovery


def save_handles(handles):
    rows = []
    for handle in handles:
        rows.append(discovery.save_handle(handle))
    return rows


def build_candidate_pool(ctx, fit_bank, sites, downstream=None):
    ranking = discovery.rank_sites(ctx, fit_bank, sites, downstream_handle=downstream)
    pool = discovery.get_candidate_pool_sites(ranking["ranked_sites"], ctx.args.candidate_pool_size, ctx.args.mass_fraction)
    return ranking, pool


def build_handles(ctx, pool, strengths, max_size, variable, graded_fit_bank, graded_cal_bank):
    handles = discovery.build_candidate_handles(pool, strengths, max_size)
    handles = discovery.add_variable_metrics(handles, graded_fit_bank, graded_cal_bank, ctx.args.graded_threshold)

    if variable == "D":
        d_handles = []
        for handle in handles:
            if handle["is_D"]:
                d_handles.append({**handle, "variable": "D"})
        return d_handles

    return handles


def filter_handles(handles, valid_singletons):
    # Remove a multi-site handle if one of its singleton subsets already passes.
    valid_site_ids = set()
    for singleton in valid_singletons:
        if singleton["k"] == 1:
            valid_site_ids.add(singleton["site_ids"][0])

    filtered = []
    for handle in handles:
        if handle["k"] == 1:
            filtered.append(handle)
            continue

        redundant = False
        for site_id in handle["site_ids"]:
            if site_id in valid_site_ids:
                redundant = True
                break

        if not redundant:
            filtered.append(handle)

    return filtered


def evaluate_round_handles(ctx, cal_bank, handles, downstream=None, r_handle=None, restore_handle=None, self_downstream=False):
    if not handles:
        return []

    if self_downstream:
        results = []
        for handle in handles:
            result = discovery.evaluate_handles(
                ctx,
                cal_bank,
                [handle],
                downstream_handle=handle,
                r_handle=r_handle,
                restore_handle=restore_handle,
            )[0]
            results.append(result)
        return results

    return discovery.evaluate_handles(
        ctx,
        cal_bank,
        handles,
        downstream_handle=downstream,
        r_handle=r_handle,
        restore_handle=restore_handle,
    )


def build_and_evaluate_handles(
    ctx,
    pool,
    strengths,
    variable,
    graded_fit_bank,
    graded_cal_bank,
    cal_bank,
    downstream=None,
    r_handle=None,
    restore_handle=None,
    self_downstream=False,
    require_restoration=False,
):
    # Evaluate singletons first, then remove pairs with a valid singleton subset.
    singletons = build_handles(ctx, pool, strengths, 1, variable, graded_fit_bank, graded_cal_bank)
    singleton_results = evaluate_round_handles(
        ctx,
        cal_bank,
        singletons,
        downstream=downstream,
        r_handle=r_handle,
        restore_handle=restore_handle,
        self_downstream=self_downstream,
    )
    valid_singletons = discovery.get_valid_handles(
        singleton_results,
        variable,
        require_restoration=require_restoration,
    )

    pairs = []
    if ctx.args.max_handle_size >= 2:
        all_handles = build_handles(ctx, pool, strengths, 2, variable, graded_fit_bank, graded_cal_bank)
        for handle in all_handles:
            if handle["k"] == 2:
                pairs.append(handle)
        pairs = filter_handles(pairs, valid_singletons)

    pair_results = evaluate_round_handles(
        ctx,
        cal_bank,
        pairs,
        downstream=downstream,
        r_handle=r_handle,
        restore_handle=restore_handle,
        self_downstream=self_downstream,
    )
    valid_pairs = discovery.get_valid_handles(
        pair_results,
        variable,
        require_restoration=require_restoration,
    )

    handles = []
    results = []
    valid = []
    handles.extend(singletons)
    handles.extend(pairs)
    results.extend(singleton_results)
    results.extend(pair_results)
    valid.extend(valid_singletons)
    valid.extend(valid_pairs)
    return handles, results, valid


def refine_handles(ctx, fit_bank, cal_bank, graded_fit_bank, graded_cal_bank, sites, late_handle, strengths, r_handle=None, require_restoration=False):
    # Freeze one handle, rebuild candidates before it, and repeat.
    variable = late_handle["variable"]
    chain = [late_handle]
    rounds = []
    current = late_handle
    used_sites = set(current["site_ids"])

    while True:
        eligible = []
        for site in sites:
            before_current = discovery.layer_order(site.site_id) < discovery.handle_order(current)
            unused = site.site_id not in used_sites
            if before_current and unused:
                eligible.append(site)

        if not eligible:
            break

        ranking, pool = build_candidate_pool(ctx, fit_bank, tuple(eligible), downstream=current)
        candidates, results, valid = build_and_evaluate_handles(
            ctx,
            pool,
            strengths,
            variable,
            graded_fit_bank,
            graded_cal_bank,
            cal_bank,
            downstream=current,
            r_handle=r_handle,
            require_restoration=require_restoration,
        )
        eligible_site_ids = []
        for site in eligible:
            eligible_site_ids.append(site.site_id)
        rounds.append({
            "downstream_handle": discovery.save_handle(current),
            "eligible_site_ids": eligible_site_ids,
            "ranking": ranking,
            "candidate_pool_sites": pool,
            "candidate_handles": save_handles(candidates),
            "results": save_handles(results),
            "valid_handles": save_handles(valid),
        })

        if not valid:
            break

        current = max(valid, key=discovery.handle_selection_key)
        current = {**current, "variable": variable}
        chain.append(current)
        for site_id in current["site_ids"]:
            used_sites.add(site_id)

    return chain, rounds


def main():
    # Discover R and D with rebuilt candidates and minimal pair handles.
    args = discovery.parse_args()
    if args.out_dir == Path("outputs/automatic_gradual_discovery_v10"):
        args.out_dir = Path("outputs/automatic_gradual_discovery_v13")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strength_values = []
    for value in args.strength_values.split(","):
        strength_values.append(float(value))
    strengths = tuple(strength_values)
    encoder = discovery.make_tinypython_encoding(args.circuit_home)
    circuit = discovery.load_candidate_circuit(args.candidate_csv, expected_count=133)
    model, model_info = discovery.load_sparse_gpt_model(model_name="csp_yolo2", circuit_home=args.circuit_home, cuda=args.cuda, flash=True, grad_checkpointing=False)
    sparse_records = discovery.convert_transformer_linears_to_sparse(model)
    site_lookup = {}
    for site in circuit.sites:
        site_lookup[site.site_id] = site
    ctx = discovery.Context(args, model, circuit.sites, site_lookup, int(encoder.encode("]\n")[0]), int(encoder.encode("]]\n")[0]), "cuda" if args.cuda else "cpu")

    coarse_examples = discovery.build_bracket_rediscovery_bank(encoder, fit_contents=48, cal_contents=24, test_contents=24, content_offset=13000)
    coarse_fit_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dfit", records_per_relation=100), "Dfit")
    coarse_cal_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dcal", records_per_relation=100), "Dcal")
    graded_examples = discovery.build_graded_d_bank(encoder, fit_contents=16, cal_contents=8, test_contents=8, content_offset=17000, q_grid=(0, 1, 2, 4))
    graded_fit_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dfit", records_per_relation=64, e_definition="active_depth"), "Dfit")
    graded_cal_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dcal", records_per_relation=48, e_definition="active_depth"), "Dcal")

    print("[1] Discover R", flush=True)
    r_ranking, r_pool = build_candidate_pool(ctx, coarse_fit_bank, ctx.sites)
    r_handles, r_cal_results, r_valid_handles = build_and_evaluate_handles(ctx, r_pool, strengths, "R", graded_fit_bank, graded_cal_bank, coarse_cal_bank)
    discovery.print_discovery_results("R", r_handles, r_cal_results, r_valid_handles, discovery.handle_order, discovery.handle_selection_key)
    assert r_valid_handles, "No R handle passed Dcal recovery"

    r_late = discovery.select_late_handle(r_valid_handles, "R")
    r_chain, r_refinement = refine_handles(ctx, coarse_fit_bank, coarse_cal_bank, graded_fit_bank, graded_cal_bank, ctx.sites, r_late, strengths)
    discovery.name_chain(r_chain, "R")
    final_r = r_chain[-1]
    discovery.print_refinement("R", r_chain, r_refinement, discovery.handle_order)
    discovery.print_frozen_handle("R", final_r, discovery.handle_order)

    print("[2] Discover D", flush=True)
    d_sites = []
    for site in ctx.sites:
        before_r = discovery.layer_order(site.site_id) < discovery.handle_order(final_r)
        outside_r = site.site_id not in final_r["weights"]
        if before_r and outside_r:
            d_sites.append(site)

    d_ranking, d_pool = build_candidate_pool(ctx, graded_fit_bank, tuple(d_sites), downstream=final_r)
    d_handles, d_cal_results, d_valid_handles = build_and_evaluate_handles(
        ctx,
        d_pool,
        strengths,
        "D",
        graded_fit_bank,
        graded_cal_bank,
        graded_cal_bank,
        r_handle=final_r,
        restore_handle=final_r,
        self_downstream=True,
        require_restoration=True,
    )
    discovery.print_discovery_results("D", d_handles, d_cal_results, d_valid_handles, discovery.handle_order, discovery.handle_selection_key)
    assert d_valid_handles, "No D handle passed Dcal recovery"

    d_late = discovery.select_late_handle(d_valid_handles, "D")
    d_chain, d_refinement = refine_handles(ctx, graded_fit_bank, graded_cal_bank, graded_fit_bank, graded_cal_bank, ctx.sites, d_late, strengths, r_handle=final_r)
    discovery.name_chain(d_chain, "D")
    final_d = d_chain[-1]
    discovery.print_refinement("D", d_chain, d_refinement, discovery.handle_order)
    discovery.print_frozen_handle("D", final_d, discovery.handle_order)

    print("[3] Final Dte certification", flush=True)
    coarse_test_bank = discovery.make_bank(ctx, coarse_examples, discovery.build_bracket_pairs(coarse_examples, split="Dte", records_per_relation=100), "Dte")
    graded_test_bank = discovery.make_bank(ctx, graded_examples, discovery.build_graded_pairs(graded_examples, split="Dte", records_per_relation=64, e_definition="active_depth"), "Dte")
    r_test_results = discovery.certify_chain(ctx, coarse_test_bank, r_chain)
    d_test_results = discovery.certify_chain(ctx, graded_test_bank, d_chain, r_handle=final_r)
    passed = True
    for row in r_test_results + d_test_results:
        if not row["edge_certified"]:
            passed = False

    d_names = []
    for row in reversed(d_chain):
        d_names.append(row["name"])
    r_names = []
    for row in reversed(r_chain):
        r_names.append(row["name"])
    final_model = "X -> " + " -> ".join(d_names + r_names) + " -> Y"

    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value

    sparse_conversion = []
    for row in sparse_records:
        sparse_conversion.append(row.to_json())

    coarse_clean_accuracy = {}
    for bank in (coarse_fit_bank, coarse_cal_bank, coarse_test_bank):
        coarse_clean_accuracy[bank.name] = discovery.clean_accuracy(bank.examples, bank.runs, split=bank.name)

    graded_clean_accuracy = {}
    for bank in (graded_fit_bank, graded_cal_bank, graded_test_bank):
        graded_clean_accuracy[bank.name] = discovery.clean_accuracy(bank.examples, bank.runs, split=bank.name)

    detailed = {
        "experiment": "automatic_gradual_discovery_v13", "config": config, "model_info": model_info, "sparse_conversion": sparse_conversion,
        "R": {"ranking": r_ranking, "candidate_pool_sites": r_pool, "candidate_handles": save_handles(r_handles), "cal_results": save_handles(r_cal_results), "valid_handles": save_handles(r_valid_handles), "refinement": r_refinement, "chain": save_handles(r_chain)},
        "D": {"ranking": d_ranking, "candidate_pool_sites": d_pool, "candidate_handles": save_handles(d_handles), "cal_results": save_handles(d_cal_results), "valid_handles": save_handles(d_valid_handles), "refinement": d_refinement, "chain": save_handles(d_chain)},
        "Dte": {"R_edges": save_handles(r_test_results), "D_edges": save_handles(d_test_results), "passed": passed},
        "banks": {"coarse": discovery.bank_manifest(coarse_examples, {"Dfit": coarse_fit_bank.pairs, "Dcal": coarse_cal_bank.pairs, "Dte": coarse_test_bank.pairs}), "graded": discovery.bank_manifest(graded_examples, {"Dfit": graded_fit_bank.pairs, "Dcal": graded_cal_bank.pairs, "Dte": graded_test_bank.pairs})},
        "clean_accuracy": {"coarse": coarse_clean_accuracy, "graded": graded_clean_accuracy},
        "final_model": final_model, "passed": passed,
    }

    handles = {}
    for row in d_chain + r_chain:
        handles[row["name"]] = discovery.short_handle(row)
    dte_edges = {}
    for row in r_test_results + d_test_results:
        dte_edges[row["edge"]] = row["edge_certified"]
    summary = {"final_model": final_model, "passed": passed, "handles": handles, "Dte_edges": dte_edges, "detailed_output": "automatic_gradual_discovery_v12_detailed.json"}

    detailed_path = args.out_dir / "automatic_gradual_discovery_v12_detailed.json"
    summary_path = args.out_dir / "automatic_gradual_discovery_v12_summary.json"
    discovery.atomic_json(detailed_path, detailed)
    discovery.atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete" if passed else "failed", "summary": str(summary_path), "details": str(detailed_path)}, indent=2))


if __name__ == "__main__":
    main()