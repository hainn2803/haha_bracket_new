from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .ablate_rediscover import (
    HandleConfiguration,
    atomic_json,
    bank_manifest,
    collect_clamped_runs,
    load_candidate_circuit,
    match_signatures,
    normalized_topk_weights,
    select_calibration_row,
)
from .graded_evidence import (
    abstract_e_signature,
    build_graded_evidence_bank,
    build_graded_pairs,
    decoded_values,
    decoder_metrics,
    e_value,
    fit_affine_decoder,
    graded_validation_summary,
)
from .progressive_rearly import (
    append_signature,
    evaluate_progressive_configurations,
    fit_binary_scalar_readout,
    load_signatures,
    mediation_summary,
    progressive_neural_signature,
    readout_accuracy,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


DEFAULT_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
E_SITE_ID = "4.attn.act_in:1249"
RMID_SITE_ID = "4.attn.resid_delta:1079"
PUBLISHED_E_WRITE = "2.attn.resid_delta:1249"


def parse_numbers(value: str, caster: Any) -> tuple[Any, ...]:
    return tuple(caster(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Characterize and progressively localize graded bracket evidence E.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--necessity-root", type=Path, default=Path("eval/openai_sparse_plot/frozen_handle_necessity_20260715"))
    parser.add_argument("--parent-rearly", type=Path, default=Path("eval/openai_sparse_plot/progressive_rearly_decoded_20260715/progressive_rearly.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/graded_evidence_20260715"))
    parser.add_argument(
        "--e-definition",
        choices=("surface_density", "active_density", "active_depth"),
        default="active_depth",
    )
    parser.add_argument("--q-grid", default="0,1,2,4")
    parser.add_argument("--fit-contents", type=int, default=16)
    parser.add_argument("--cal-contents", type=int, default=8)
    parser.add_argument("--test-contents", type=int, default=8)
    parser.add_argument("--content-offset", type=int, default=17000)
    parser.add_argument("--fit-records-per-relation", type=int, default=64)
    parser.add_argument("--cal-records-per-relation", type=int, default=48)
    parser.add_argument("--test-records-per-relation", type=int, default=64)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _states(margins: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(margins) > 0.0, 1, -1)


def _summary_from_arrays(
    pairs: Any,
    examples_by_id: Any,
    runs: Any,
    margins: np.ndarray,
    probes: np.ndarray,
    *,
    e_definition: str,
    e_decoder: Any,
    rmid_readout: Any,
) -> dict[str, Any]:
    clean_e = {
        example_id: e_decoder.predict(run.features_by_site[E_SITE_ID]) for example_id, run in runs.items()
    }
    abstract_e = {
        example_id: e_value(examples_by_id[example_id], e_definition) for example_id in runs
    }
    patched_e = decoded_values(probes, e_decoder, 0)
    patched_rmid = [rmid_readout.predict(value) for value in probes[:, 1]]
    return graded_validation_summary(
        pairs,
        examples_by_id,
        abstract_e,
        clean_e,
        patched_e,
        patched_rmid,
        _states(margins),
    )


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Graded Evidence E -> R_mid",
        "",
        f"- abstract E definition: `{payload['manifest']['e_definition']}`",
        f"- frozen E handle: `{E_SITE_ID}`",
        f"- frozen R_mid: `{RMID_SITE_ID}`",
        "- full audit universe: `133`; primary upstream universe excludes only the frozen E handle",
        "",
        "## Representation",
        "",
        f"- E decoder Dcal R2 / Pearson: `{payload['decoder_metrics']['Dcal']['r2']:.3f}` / `{payload['decoder_metrics']['Dcal']['pearson']:.3f}`",
        f"- E decoder Dte R2 / Pearson: `{payload['decoder_metrics']['Dte']['r2']:.3f}` / `{payload['decoder_metrics']['Dte']['pearson']:.3f}`",
        f"- E threshold Rmid Dte accuracy: `{payload['E_threshold_accuracy']['Dte']:.3f}`",
        "",
        "## Direct E Handle",
        "",
        f"- heldout pass: `{payload['direct_E_handle']['summary']['passes']}`",
        f"- Rmid mediation pass: `{payload['direct_E_handle']['mediation']['passes']}`",
        "",
        "## Progressive Upstream PLOT",
        "",
        f"- selected support: `{', '.join(payload['upstream']['calibration']['best']['site_ids'])}`",
        f"- Dte pass: `{payload['upstream']['heldout']['summary']['passes']}`",
        f"- published `{PUBLISHED_E_WRITE}` rank: `{payload['posthoc_ranks'][PUBLISHED_E_WRITE]}`",
        f"- full E -> R_mid model accepted: `{payload['model_accepted']}`",
        "",
        "## Conclusion",
        "",
        payload["conclusion"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parent = json.loads(args.parent_rearly.read_text(encoding="utf-8"))
    if not parent.get("R_early_accepted") or parent["calibration"]["best"]["site_ids"] != [E_SITE_ID]:
        raise RuntimeError("graded E stage requires the accepted frozen parent handle 4.attn.act_in:1249")
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=args.expected_node_count)
    site_lookup = {site.site_id: site for site in circuit.sites}
    enc = make_tinypython_encoding(args.circuit_home)
    q_grid = parse_numbers(args.q_grid, int)
    examples = build_graded_evidence_bank(
        enc,
        fit_contents=args.fit_contents,
        cal_contents=args.cal_contents,
        test_contents=args.test_contents,
        content_offset=args.content_offset,
        q_grid=q_grid,
    )
    pairs = {
        "Dfit": build_graded_pairs(examples, split="Dfit", records_per_relation=args.fit_records_per_relation, e_definition=args.e_definition),
        "Dcal": build_graded_pairs(examples, split="Dcal", records_per_relation=args.cal_records_per_relation, e_definition=args.e_definition),
        "Dte": build_graded_pairs(examples, split="Dte", records_per_relation=args.test_records_per_relation, e_definition=args.e_definition),
    }
    bank = bank_manifest(examples, pairs)
    if not bank["content_splits_disjoint"]:
        raise RuntimeError("graded data splits overlap")
    examples_by_id = {row.example_id: row for row in examples}
    means = torch.load(args.necessity_root / "bracket" / "Dfit_task_means.pt", map_location="cpu", weights_only=True)
    negative = int(enc.encode("]\n")[0])
    positive = int(enc.encode("]]\n")[0])
    model, model_info = load_sparse_gpt_model(
        model_name="csp_yolo2",
        circuit_home=args.circuit_home,
        cuda=bool(args.cuda),
        flash=True,
        grad_checkpointing=False,
    )
    sparse_records = convert_transformer_linears_to_sparse(model)
    device = "cuda" if args.cuda else "cpu"
    runs = collect_clamped_runs(
        model,
        examples,
        candidate_sites=circuit.sites,
        disabled_sites=(),
        hook_means=means,
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    clean_accuracy_by_split = {
        split: float(
            np.mean(
                [
                    runs[row.example_id].predicted_value == row.variable_value
                    for row in examples
                    if row.split == split
                ]
            )
        )
        for split in ("Dfit", "Dcal", "Dte")
    }
    if min(clean_accuracy_by_split.values()) < 0.90:
        raise RuntimeError(f"graded padding bank is not cleanly solved: {clean_accuracy_by_split}")
    fit_rows = [row for row in examples if row.split == "Dfit"]
    e_decoder = fit_affine_decoder(
        [runs[row.example_id].features_by_site[E_SITE_ID] for row in fit_rows],
        [e_value(row, args.e_definition) for row in fit_rows],
    )
    decoder_scores = {
        split: decoder_metrics(
            e_decoder,
            [runs[row.example_id].features_by_site[E_SITE_ID] for row in examples if row.split == split],
            [e_value(row, args.e_definition) for row in examples if row.split == split],
        )
        for split in ("Dfit", "Dcal", "Dte")
    }
    rmid_readout = fit_binary_scalar_readout(
        [runs[row.example_id].features_by_site[RMID_SITE_ID] for row in fit_rows],
        [row.variable_value for row in fit_rows],
    )
    rmid_accuracy = {
        split: readout_accuracy(rmid_readout, examples, runs, split=split, site_id=RMID_SITE_ID)
        for split in ("Dfit", "Dcal", "Dte")
    }
    e_threshold = fit_binary_scalar_readout(
        [e_decoder.predict(runs[row.example_id].features_by_site[E_SITE_ID]) for row in fit_rows],
        [row.variable_value for row in fit_rows],
    )
    e_threshold_accuracy = {
        split: float(
            np.mean(
                [
                    e_threshold.predict(e_decoder.predict(runs[row.example_id].features_by_site[E_SITE_ID]))
                    == row.variable_value
                    for row in examples
                    if row.split == split
                ]
            )
        )
        for split in ("Dfit", "Dcal", "Dte")
    }
    manifest = {
        "experiment": "graded_evidence_E_to_Rmid",
        "e_definition": args.e_definition,
        "q_grid": list(q_grid),
        "candidate_count_audit": 133,
        "candidate_count_primary": 132,
        "primary_exclusion": E_SITE_ID,
        "known_published_site_used_for_selection": False,
        "signature": "abstract E(source)-E(base); neural frozen decoded E_handle(y_swap)-E_handle(y_base)",
        "Dfit_used_for": "affine E decoder, binary readouts, and PLOT signatures",
        "Dcal_used_for": "top-K and strength calibration",
        "Dte_used_for": "heldout validation and mediation only",
        "bank": bank,
        "clean_accuracy": clean_accuracy_by_split,
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    # First validate the frozen E handle itself.
    direct = HandleConfiguration("direct_E", {E_SITE_ID: 1.0}, 1.0)
    direct_margins, direct_probes = evaluate_progressive_configurations(
        model,
        (direct,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[E_SITE_ID], site_lookup[RMID_SITE_ID]),
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    direct_summary = _summary_from_arrays(
        pairs["Dte"], examples_by_id, runs, direct_margins[0], direct_probes[0], e_definition=args.e_definition, e_decoder=e_decoder, rmid_readout=rmid_readout
    )
    restored_rmid_margins, _ = evaluate_progressive_configurations(
        model,
        (direct,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[RMID_SITE_ID],),
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
        restore_probe_site_ids=(RMID_SITE_ID,),
    )
    direct_mediation = mediation_summary(
        pairs["Dte"], examples_by_id, runs, direct_margins[0], restored_rmid_margins[0]
    )
    # Progressive PLOT upstream of the frozen E handle.
    signature_path = args.out_dir / "Dfit_upstream_signatures.jsonl"
    signatures = load_signatures(signature_path)
    for start in range(0, len(circuit.sites), int(args.candidate_chunk_size)):
        chunk = [site for site in circuit.sites[start : start + int(args.candidate_chunk_size)] if site.site_id not in signatures]
        if not chunk:
            continue
        configs = tuple(HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in chunk)
        margins, probes = evaluate_progressive_configurations(
            model,
            configs,
            pairs["Dfit"],
            examples=examples_by_id,
            runs=runs,
            site_lookup=site_lookup,
            probe_sites=(site_lookup[E_SITE_ID],),
            negative_token_id=negative,
            positive_token_id=positive,
            device=device,
            max_batch_size=args.max_batch_size,
        )
        for index, site in enumerate(chunk):
            signature = progressive_neural_signature(
                pairs["Dfit"],
                runs,
                margins[index],
                probes[index],
                probe_site_ids=(E_SITE_ID,),
                probe_scales=(e_decoder.slope,),
                include_output=False,
            )
            signatures[site.site_id] = signature
            append_signature(signature_path, site.site_id, signature)
        print(f"graded E signatures {len(signatures)}/133", flush=True)
    abstract = abstract_e_signature(pairs["Dfit"], examples_by_id, definition=args.e_definition)
    audit_selector = match_signatures(abstract, {site_id: signatures[site_id] for site_id in circuit.node_ids}, epsilon=args.selector_epsilon, beta=args.selector_beta)
    primary_ids = tuple(site_id for site_id in circuit.node_ids if site_id != E_SITE_ID)
    primary_selector = match_signatures(abstract, {site_id: signatures[site_id] for site_id in primary_ids}, epsilon=args.selector_epsilon, beta=args.selector_beta)
    atomic_json(args.out_dir / "selector_all133.json", audit_selector)
    atomic_json(args.out_dir / "selector_primary132.json", primary_selector)
    configs = []
    metas = []
    for k in parse_numbers(args.k_grid, int):
        weights = normalized_topk_weights(primary_selector["ranked"], k)
        for strength in parse_numbers(args.strength_grid, float):
            configs.append(HandleConfiguration(f"K{k}_lambda{strength:g}", weights, strength))
            metas.append({"handle_id": f"K{k}_lambda{strength:g}", "k": k, "strength": strength, "weights": weights})
    cal_margins, cal_probes = evaluate_progressive_configurations(
        model,
        configs,
        pairs["Dcal"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[E_SITE_ID], site_lookup[RMID_SITE_ID]),
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    grid = []
    for index, meta in enumerate(metas):
        grid.append(
            {
                **meta,
                "site_ids": list(meta["weights"]),
                "summary": _summary_from_arrays(
                    pairs["Dcal"], examples_by_id, runs, cal_margins[index], cal_probes[index], e_definition=args.e_definition, e_decoder=e_decoder, rmid_readout=rmid_readout
                ),
            }
        )
    best = dict(select_calibration_row(grid))
    selected = HandleConfiguration("selected", best["weights"], best["strength"])
    test_margins, test_probes = evaluate_progressive_configurations(
        model,
        (selected,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[E_SITE_ID], site_lookup[RMID_SITE_ID]),
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    heldout_summary = _summary_from_arrays(
        pairs["Dte"], examples_by_id, runs, test_margins[0], test_probes[0], e_definition=args.e_definition, e_decoder=e_decoder, rmid_readout=rmid_readout
    )
    restored_e_margins, _ = evaluate_progressive_configurations(
        model,
        (selected,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[E_SITE_ID], site_lookup[RMID_SITE_ID]),
        negative_token_id=negative,
        positive_token_id=positive,
        device=device,
        max_batch_size=args.max_batch_size,
        restore_probe_site_ids=(E_SITE_ID,),
    )
    upstream_mediation = mediation_summary(
        pairs["Dte"], examples_by_id, runs, test_margins[0], restored_e_margins[0]
    )
    audit_rank = {row["site_id"]: index + 1 for index, row in enumerate(audit_selector["ranked"])}
    model_accepted = bool(
        decoder_scores["Dcal"]["pearson"] >= 0.90
        and decoder_scores["Dte"]["pearson"] >= 0.90
        and e_threshold_accuracy["Dcal"] >= 0.90
        and e_threshold_accuracy["Dte"] >= 0.90
        and direct_summary["passes"]
        and direct_mediation["passes"]
    )
    variable_label = "active-depth D" if args.e_definition == "active_depth" else f"graded E ({args.e_definition})"
    conclusion = (
        f"The frozen 1249 input is a validated {variable_label} variable whose effect on output is mediated by binary R_mid."
        if model_accepted
        else f"The tested {args.e_definition} definition does not fully validate a graded E -> R_mid abstraction on fresh data."
    )
    result = {
        "manifest": manifest,
        "E_decoder": e_decoder.to_dict(),
        "decoder_metrics": decoder_scores,
        "Rmid_readout_accuracy": rmid_accuracy,
        "E_threshold_accuracy": e_threshold_accuracy,
        "direct_E_handle": {"summary": direct_summary, "mediation": direct_mediation},
        "upstream": {
            "calibration": {"grid": grid, "best": best},
            "heldout": {"summary": heldout_summary, "mediation": upstream_mediation},
        },
        "posthoc_ranks": {site_id: audit_rank.get(site_id) for site_id in (E_SITE_ID, RMID_SITE_ID, PUBLISHED_E_WRITE)},
        "model_accepted": model_accepted,
        "conclusion": conclusion,
    }
    atomic_json(args.out_dir / "graded_evidence.json", result)
    _write_report(args.out_dir / "graded_evidence.md", result)
    print(json.dumps({"status": "complete", "model_accepted": model_accepted, "decoder": decoder_scores, "direct": result["direct_E_handle"], "upstream_best": best, "upstream_heldout": result["upstream"]["heldout"], "posthoc_ranks": result["posthoc_ranks"]}, indent=2))


if __name__ == "__main__":
    main()
