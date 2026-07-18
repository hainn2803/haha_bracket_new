from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .ablate_rediscover import (
    HandleConfiguration,
    atomic_json,
    bank_manifest,
    build_bracket_pairs,
    build_bracket_rediscovery_bank,
    collect_clamped_runs,
    load_candidate_circuit,
    match_signatures,
    normalized_topk_weights,
    select_calibration_row,
)
from .progressive_rearly import (
    append_signature,
    evaluate_progressive_configurations,
    fit_binary_scalar_readout,
    load_signatures,
    mediation_summary,
    progressive_abstract_signature,
    progressive_neural_signature,
    progressive_relation_summary,
    readout_accuracy,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


DEFAULT_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_OUT = Path("eval/openai_sparse_plot/progressive_rearly_20260715")
RMID_SITE_ID = "4.attn.resid_delta:1079"
POSTHOC_SITE_IDS = (
    "2.attn.resid_delta:1249",
    RMID_SITE_ID,
    "7.mlp.act_in:1079",
    "7.mlp.post_act:4133",
    "7.mlp.resid_delta:2041",
)


def parse_numbers(value: str, caster: Any) -> tuple[Any, ...]:
    return tuple(caster(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive all-site PLOT search upstream of certified R_mid.")
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--necessity-root", type=Path, default=Path("eval/openai_sparse_plot/frozen_handle_necessity_20260715"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fit-contents", type=int, default=24)
    parser.add_argument("--cal-contents", type=int, default=12)
    parser.add_argument("--test-contents", type=int, default=12)
    parser.add_argument("--content-offset", type=int, default=14000)
    parser.add_argument("--fit-records-per-relation", type=int, default=48)
    parser.add_argument("--cal-records-per-relation", type=int, default=32)
    parser.add_argument("--test-records-per-relation", type=int, default=48)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--signature-mode", choices=("augmented", "rmid_only"), default="augmented")
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _report(path: Path, payload: Mapping[str, Any]) -> None:
    selected = payload["calibration"]["best"]
    heldout = payload["heldout"]
    lines = [
        "# Progressive R_early Search",
        "",
        "- full audit universe: `133` localized singleton sites",
        "- primary universe: `132` sites; only the frozen downstream R_mid site is excluded",
        f"- signature mode: `{payload['manifest']['signature_mode']}`",
        f"- neural signature per pair: `{payload['manifest']['signature']}`",
        f"- abstract signature per pair: `{payload['manifest']['abstract_signature']}`",
        "- selector: raw cosine-cost one-sided UOT",
        "",
        "## Selection",
        "",
        f"- selected sites: `{', '.join(selected['site_ids'])}`",
        f"- K / strength: `{selected['k']}` / `{selected['strength']}`",
        f"- Dcal score: `{selected['summary']['score']:.3f}`",
        f"- Dte score: `{heldout['summary']['score']:.3f}`",
        f"- mediation passes: `{heldout['mediation']['passes']}`",
        f"- R_early accepted: `{payload['R_early_accepted']}`",
        "",
        "## Post-hoc Ranks",
        "",
    ]
    for site_id, ranks in payload["posthoc_ranks"].items():
        lines.append(f"- `{site_id}`: primary `{ranks['primary']}`, audit `{ranks['audit']}`")
    lines.extend(["", "## Conclusion", "", payload["conclusion"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=args.expected_node_count)
    site_lookup = {site.site_id: site for site in circuit.sites}
    if RMID_SITE_ID not in site_lookup:
        raise RuntimeError("certified R_mid is absent from the full localized circuit")
    enc = make_tinypython_encoding(args.circuit_home)
    examples = build_bracket_rediscovery_bank(
        enc,
        fit_contents=args.fit_contents,
        cal_contents=args.cal_contents,
        test_contents=args.test_contents,
        content_offset=args.content_offset,
    )
    pairs = {
        "Dfit": build_bracket_pairs(examples, split="Dfit", records_per_relation=args.fit_records_per_relation),
        "Dcal": build_bracket_pairs(examples, split="Dcal", records_per_relation=args.cal_records_per_relation),
        "Dte": build_bracket_pairs(examples, split="Dte", records_per_relation=args.test_records_per_relation),
    }
    bank = bank_manifest(examples, pairs)
    if not bank["content_splits_disjoint"]:
        raise RuntimeError("data splits overlap")
    examples_by_id = {row.example_id: row for row in examples}
    means = torch.load(
        args.necessity_root / "bracket" / "Dfit_task_means.pt",
        map_location="cpu",
        weights_only=True,
    )
    negative_token_id = int(enc.encode("]\n")[0])
    positive_token_id = int(enc.encode("]]\n")[0])
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
        negative_token_id=negative_token_id,
        positive_token_id=positive_token_id,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    fit_rows = [row for row in examples if row.split == "Dfit"]
    readout = fit_binary_scalar_readout(
        [runs[row.example_id].features_by_site[RMID_SITE_ID] for row in fit_rows],
        [row.variable_value for row in fit_rows],
    )
    readout_scores = {
        split: readout_accuracy(readout, examples, runs, split=split, site_id=RMID_SITE_ID)
        for split in ("Dfit", "Dcal", "Dte")
    }
    if min(readout_scores.values()) < 0.90:
        raise RuntimeError(f"certified R_mid scalar readout is not valid on fresh data: {readout_scores}")
    manifest = {
        "experiment": "progressive_rearly",
        "candidate_count_audit": len(circuit.sites),
        "candidate_count_primary": len(circuit.sites) - 1,
        "candidate_csv_sha256": circuit.csv_sha256,
        "candidate_filtering_audit": "none",
        "candidate_filtering_primary": f"only frozen downstream handle {RMID_SITE_ID} excluded",
        "known_sites_used_for_selection": False,
        "signature_mode": args.signature_mode,
        "signature": (
            "per pair [delta frozen decoded R_mid score, delta output margin]"
            if args.signature_mode == "augmented"
            else "per pair [delta frozen decoded R_mid score]"
        ),
        "abstract_signature": (
            "per pair [delta R, delta R]"
            if args.signature_mode == "augmented"
            else "per pair [delta R]"
        ),
        "followup_rationale": (
            None
            if args.signature_mode == "augmented"
            else "progressive PLOT targets only the previously certified downstream R_mid; output remains a calibration and heldout gate"
        ),
        "Dfit_used_for": "R_mid scalar readout fit and PLOT signatures",
        "Dcal_used_for": "top-K and strength calibration",
        "calibration_score": (
            "unweighted mean of sensitivity-output, sensitivity-Rmid-state, "
            "aggregate-invariance-output, and aggregate-invariance-Rmid-state"
        ),
        "Dte_used_for": "final heldout and mediation only",
        "bank": bank,
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "Rmid_readout": {**readout.to_dict(), "accuracies": readout_scores},
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    signature_path = args.out_dir / "Dfit_signatures.jsonl"
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
            probe_sites=(site_lookup[RMID_SITE_ID],),
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
            device=device,
            max_batch_size=args.max_batch_size,
        )
        for index, site in enumerate(chunk):
            signature = progressive_neural_signature(
                pairs["Dfit"],
                runs,
                margins[index],
                probes[index],
                probe_site_ids=(RMID_SITE_ID,),
                probe_scales=(float(readout.orientation),),
                include_output=args.signature_mode == "augmented",
            )
            signatures[site.site_id] = signature
            append_signature(signature_path, site.site_id, signature)
        print(f"R_early signatures {len(signatures)}/{len(circuit.sites)}", flush=True)
    if set(signatures) != set(circuit.node_ids):
        raise RuntimeError("Dfit signatures do not cover all 133 audit sites")
    abstract = progressive_abstract_signature(
        pairs["Dfit"],
        examples_by_id,
        components=2 if args.signature_mode == "augmented" else 1,
    )
    audit_selector = match_signatures(
        abstract,
        {site_id: signatures[site_id] for site_id in circuit.node_ids},
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    primary_ids = tuple(site_id for site_id in circuit.node_ids if site_id != RMID_SITE_ID)
    primary_selector = match_signatures(
        abstract,
        {site_id: signatures[site_id] for site_id in primary_ids},
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    atomic_json(args.out_dir / "selector_audit_all133.json", audit_selector)
    atomic_json(args.out_dir / "selector_primary_132.json", primary_selector)
    k_grid = parse_numbers(args.k_grid, int)
    strength_grid = parse_numbers(args.strength_grid, float)
    configs = []
    metas = []
    for k in k_grid:
        weights = normalized_topk_weights(primary_selector["ranked"], int(k))
        for strength in strength_grid:
            handle_id = f"K{k}_lambda{strength:g}"
            configs.append(HandleConfiguration(handle_id, weights, float(strength)))
            metas.append({"handle_id": handle_id, "k": int(k), "strength": float(strength), "weights": weights})
    cal_margins, cal_probes = evaluate_progressive_configurations(
        model,
        configs,
        pairs["Dcal"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[RMID_SITE_ID],),
        negative_token_id=negative_token_id,
        positive_token_id=positive_token_id,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    grid = []
    for index, meta in enumerate(metas):
        grid.append(
            {
                **meta,
                "site_ids": list(meta["weights"]),
                "summary": progressive_relation_summary(
                    pairs["Dcal"],
                    examples_by_id,
                    runs,
                    cal_margins[index],
                    cal_probes[index],
                    probe_site_id=RMID_SITE_ID,
                    readout=readout,
                ),
            }
        )
    best = dict(select_calibration_row(grid))
    selected = HandleConfiguration("selected", best["weights"], float(best["strength"]))
    test_margins, test_probes = evaluate_progressive_configurations(
        model,
        (selected,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[RMID_SITE_ID],),
        negative_token_id=negative_token_id,
        positive_token_id=positive_token_id,
        device=device,
        max_batch_size=args.max_batch_size,
    )
    restored_margins, _restored_probes = evaluate_progressive_configurations(
        model,
        (selected,),
        pairs["Dte"],
        examples=examples_by_id,
        runs=runs,
        site_lookup=site_lookup,
        probe_sites=(site_lookup[RMID_SITE_ID],),
        negative_token_id=negative_token_id,
        positive_token_id=positive_token_id,
        device=device,
        max_batch_size=args.max_batch_size,
        restore_probes_to_base=True,
    )
    heldout_summary = progressive_relation_summary(
        pairs["Dte"],
        examples_by_id,
        runs,
        test_margins[0],
        test_probes[0],
        probe_site_id=RMID_SITE_ID,
        readout=readout,
    )
    mediation = mediation_summary(
        pairs["Dte"], examples_by_id, runs, test_margins[0], restored_margins[0]
    )
    accepted = bool(heldout_summary["all_required_rates_at_least_0_90"] and mediation["passes"])
    primary_rank = {row["site_id"]: index + 1 for index, row in enumerate(primary_selector["ranked"])}
    audit_rank = {row["site_id"]: index + 1 for index, row in enumerate(audit_selector["ranked"])}
    posthoc = {
        site_id: {"primary": primary_rank.get(site_id), "audit": audit_rank.get(site_id)}
        for site_id in POSTHOC_SITE_IDS
    }
    conclusion = (
        "A distinct upstream binary R_early handle was accepted and its output effect is mediated by the certified R_mid."
        if accepted
        else "No distinct R_early handle passed both heldout state/output validation and the frozen R_mid restoration test."
    )
    result = {
        "manifest": manifest,
        "calibration": {"grid": grid, "best": best},
        "heldout": {"summary": heldout_summary, "mediation": mediation},
        "posthoc_ranks": posthoc,
        "R_early_accepted": accepted,
        "conclusion": conclusion,
    }
    atomic_json(args.out_dir / "progressive_rearly.json", result)
    _report(args.out_dir / "progressive_rearly.md", result)
    print(json.dumps({"status": "complete", "accepted": accepted, "best": best, "heldout": result["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
