from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .ablate_rediscover import (
    HandleConfiguration,
    abstract_signature,
    atomic_json,
    bank_manifest,
    build_bracket_pairs,
    build_bracket_rediscovery_bank,
    build_quote_pairs,
    build_quote_rediscovery_bank,
    can_certify_redundancy,
    clean_accuracy,
    collect_clamped_runs,
    evaluate_configurations,
    load_candidate_circuit,
    match_signatures,
    normalized_topk_weights,
    relation_summary,
    select_calibration_row,
    write_jsonl,
)
from .activation import ChannelSite
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


DEFAULT_CSVS = {
    "quote": Path("eval/openai_sparse_plot/string_closing_prune_v2_64/string_closing_circuit_nodes.csv"),
    "bracket": Path(
        "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
    ),
}
INITIAL_DISABLED = {
    "quote": ("0.mlp.resid_delta:460",),
    "bracket": ("4.attn.resid_delta:1079",),
}


def parse_csv_numbers(value: str, caster: Any) -> tuple[Any, ...]:
    return tuple(caster(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean-ablate a certified handle and rerun raw-output PLOT.")
    parser.add_argument("--task", choices=("quote", "bracket"), required=True)
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--necessity-root", type=Path, default=Path("eval/openai_sparse_plot/frozen_handle_necessity_20260715"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/ablate_rediscover_20260715"))
    parser.add_argument("--fit-contents", type=int, default=24)
    parser.add_argument("--cal-contents", type=int, default=12)
    parser.add_argument("--test-contents", type=int, default=12)
    parser.add_argument("--content-offset", type=int, default=13000)
    parser.add_argument("--fit-records-per-relation", type=int, default=48)
    parser.add_argument("--cal-records-per-relation", type=int, default=32)
    parser.add_argument("--test-records-per-relation", type=int, default=48)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _token_ids(task: str, enc: Any) -> tuple[int, int]:
    if task == "quote":
        tokens = quote_token_ids(enc)
        return int(tokens["single"]), int(tokens["double"])
    return int(enc.encode("]\n")[0]), int(enc.encode("]]\n")[0])


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [f"# Ablate And Rediscover: {payload['task'].title()}", ""]
    lines.extend(
        [
            f"- full localized candidate universe: `{payload['candidate_count']}`",
            "- candidate filtering: only sites explicitly disabled by preceding rounds are removed from the primary search",
            "- signature: raw `phi(y_swap) - phi(y_base)` output-margin effects",
            "- selector: raw cosine-cost one-sided UOT",
            "",
        ]
    )
    for row in payload["rounds"]:
        lines.extend(
            [
                f"## Round {row['round']}",
                "",
                f"- disabled before search: `{', '.join(row['disabled_before'])}`",
                f"- remaining candidates: `{row['candidate_count']}`",
                f"- clamped Dte clean accuracy: `{row['clean_accuracy']['Dte']:.3f}`",
                f"- selected handle: `{', '.join(row['selected_site_ids'])}`",
                f"- Dcal score: `{row['calibration_best']['summary']['score']:.3f}`",
                f"- Dte rates: `{json.dumps(row['heldout']['rates'], sort_keys=True)}`",
                f"- diagnostic behavioral pass: `{row['diagnostic_handle_pass']}`",
                f"- certified natural redundancy: `{row['redundancy_certified']}`",
                "",
            ]
        )
    lines.extend(["## Conclusion", "", f"{payload['conclusion']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    task = str(args.task)
    model_name = "csp_yolo1" if task == "quote" else "csp_yolo2"
    expected_count = 64 if task == "quote" else 133
    task_dir = args.out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_candidate_circuit(args.candidate_csv or DEFAULT_CSVS[task], expected_count=expected_count)
    site_lookup = {site.site_id: site for site in circuit.sites}
    enc = make_tinypython_encoding(args.circuit_home)
    examples = (
        build_quote_rediscovery_bank(
            enc,
            fit_contents=args.fit_contents,
            cal_contents=args.cal_contents,
            test_contents=args.test_contents,
            content_offset=args.content_offset,
        )
        if task == "quote"
        else build_bracket_rediscovery_bank(
            enc,
            fit_contents=args.fit_contents,
            cal_contents=args.cal_contents,
            test_contents=args.test_contents,
            content_offset=args.content_offset,
        )
    )
    pair_builder = build_quote_pairs if task == "quote" else build_bracket_pairs
    pairs_by_split = {
        "Dfit": pair_builder(examples, split="Dfit", records_per_relation=args.fit_records_per_relation),
        "Dcal": pair_builder(examples, split="Dcal", records_per_relation=args.cal_records_per_relation),
        "Dte": pair_builder(examples, split="Dte", records_per_relation=args.test_records_per_relation),
    }
    bank = bank_manifest(examples, pairs_by_split)
    if not bank["content_splits_disjoint"]:
        raise RuntimeError("Dfit, Dcal, and Dte contents overlap")
    examples_by_id = {row.example_id: row for row in examples}
    means_path = args.necessity_root / task / "Dfit_task_means.pt"
    if not means_path.exists():
        raise FileNotFoundError(f"missing frozen Dfit task means: {means_path}")
    hook_means = torch.load(means_path, map_location="cpu", weights_only=True)
    negative_token_id, positive_token_id = _token_ids(task, enc)
    model, model_info = load_sparse_gpt_model(
        model_name=model_name,
        circuit_home=args.circuit_home,
        cuda=bool(args.cuda),
        flash=True,
        grad_checkpointing=False,
    )
    sparse_records = convert_transformer_linears_to_sparse(model)
    atomic_json(
        task_dir / "manifest.json",
        {
            "experiment": "ablate_and_rediscover",
            "task": task,
            "model": model_name,
            "candidate_count": len(circuit.sites),
            "candidate_csv": circuit.csv_path,
            "candidate_csv_sha256": circuit.csv_sha256,
            "candidate_filtering": "none, except explicitly disabled handles in each primary round",
            "initial_disabled": list(INITIAL_DISABLED[task]),
            "mean_ablation": "unconditional all-token Dfit task mean frozen by the necessity audit",
            "signature": "abstract variable(source)-variable(base); neural phi(y_swap)-phi(y_base)",
            "phi": "positive-class logit minus negative-class logit",
            "Dfit_used_for": "signature matching only",
            "Dcal_used_for": "top-K and strength calibration only",
            "Dte_used_for": "one final heldout evaluation only",
            "bank": bank,
            "model_info": model_info,
            "sparse_conversion": [row.to_json() for row in sparse_records],
        },
    )
    k_grid = parse_csv_numbers(args.k_grid, int)
    strength_grid = parse_csv_numbers(args.strength_grid, float)
    device = "cuda" if args.cuda else "cpu"
    disabled_ids = list(INITIAL_DISABLED[task])
    rounds: list[dict[str, Any]] = []
    for round_index in range(int(args.max_rounds)):
        disabled_sites = tuple(site_lookup[site_id] for site_id in disabled_ids)
        candidate_sites = tuple(site for site in circuit.sites if site.site_id not in set(disabled_ids))
        if not candidate_sites:
            break
        runs = collect_clamped_runs(
            model,
            examples,
            candidate_sites=circuit.sites,
            disabled_sites=disabled_sites,
            hook_means=hook_means,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
            device=device,
            max_batch_size=args.max_batch_size,
        )
        clean = {split: clean_accuracy(examples, runs, split=split) for split in ("Dfit", "Dcal", "Dte")}
        singleton_configs = tuple(
            HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0) for site in candidate_sites
        )
        fit_margins = evaluate_configurations(
            model,
            singleton_configs,
            pairs_by_split["Dfit"],
            examples=examples_by_id,
            runs=runs,
            site_lookup=site_lookup,
            disabled_sites=disabled_sites,
            hook_means=hook_means,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
            device=device,
            max_batch_size=args.max_batch_size,
        )
        neural_by_site = {
            config.handle_id: tuple(
                float(fit_margins[index, pair_index] - runs[pair.base_id].class_margin)
                for pair_index, pair in enumerate(pairs_by_split["Dfit"])
            )
            for index, config in enumerate(singleton_configs)
        }
        abstract = abstract_signature(pairs_by_split["Dfit"], examples_by_id)
        selector = match_signatures(
            abstract,
            neural_by_site,
            epsilon=args.selector_epsilon,
            beta=args.selector_beta,
        )
        round_dir = task_dir / f"round_{round_index}"
        write_jsonl(
            round_dir / "Dfit_signatures.jsonl",
            (
                {"site_id": site_id, "signature": list(signature)}
                for site_id, signature in neural_by_site.items()
            ),
        )
        atomic_json(round_dir / "selector.json", {**selector, "abstract_signature": list(abstract)})
        calibration_configs: list[HandleConfiguration] = []
        calibration_meta: list[dict[str, Any]] = []
        for k in k_grid:
            if int(k) > len(candidate_sites):
                continue
            weights = normalized_topk_weights(selector["ranked"], int(k))
            for strength in strength_grid:
                handle_id = f"K{k}_lambda{strength:g}"
                calibration_configs.append(HandleConfiguration(handle_id, weights, float(strength)))
                calibration_meta.append(
                    {"handle_id": handle_id, "k": int(k), "strength": float(strength), "weights": weights}
                )
        cal_margins = evaluate_configurations(
            model,
            calibration_configs,
            pairs_by_split["Dcal"],
            examples=examples_by_id,
            runs=runs,
            site_lookup=site_lookup,
            disabled_sites=disabled_sites,
            hook_means=hook_means,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
            device=device,
            max_batch_size=args.max_batch_size,
        )
        calibration_rows = []
        for index, meta in enumerate(calibration_meta):
            calibration_rows.append(
                {
                    **meta,
                    "summary": relation_summary(
                        pairs_by_split["Dcal"], examples_by_id, cal_margins[index]
                    ),
                }
            )
        best = dict(select_calibration_row(calibration_rows))
        selected = HandleConfiguration(
            "selected",
            {str(key): float(value) for key, value in best["weights"].items()},
            float(best["strength"]),
        )
        heldout_margins = evaluate_configurations(
            model,
            (selected,),
            pairs_by_split["Dte"],
            examples=examples_by_id,
            runs=runs,
            site_lookup=site_lookup,
            disabled_sites=disabled_sites,
            hook_means=hook_means,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
            device=device,
            max_batch_size=args.max_batch_size,
        )[0]
        heldout = relation_summary(pairs_by_split["Dte"], examples_by_id, heldout_margins)
        diagnostic_pass = bool(heldout["all_rates_at_least_0_90"])
        redundancy = can_certify_redundancy(clean_dte_accuracy=clean["Dte"], heldout_summary=heldout)
        round_payload = {
            "round": round_index,
            "disabled_before": list(disabled_ids),
            "candidate_count": len(candidate_sites),
            "clean_accuracy": clean,
            "calibration_best": best,
            "selected_site_ids": list(selected.weights_by_site),
            "heldout": heldout,
            "diagnostic_handle_pass": diagnostic_pass,
            "redundancy_certified": redundancy,
        }
        rounds.append(round_payload)
        atomic_json(round_dir / "calibration.json", {"grid": calibration_rows, "best": best})
        atomic_json(round_dir / "heldout.json", round_payload)
        print(json.dumps(round_payload, indent=2), flush=True)
        if not redundancy:
            break
        disabled_ids.extend(site_id for site_id in selected.weights_by_site if site_id not in disabled_ids)
    if rounds and rounds[-1]["redundancy_certified"]:
        conclusion = "At least one naturally redundant alternative handle was certified after ablation."
    elif rounds and rounds[-1]["diagnostic_handle_pass"]:
        conclusion = (
            "A diagnostic intervention handle was found, but natural redundancy was not certified because the "
            "clamped model did not retain at least 0.90 clean accuracy."
        )
    else:
        conclusion = "No behaviorally valid alternative handle was certified after ablating the learned handle."
    result = {
        "task": task,
        "candidate_count": len(circuit.sites),
        "rounds": rounds,
        "conclusion": conclusion,
    }
    atomic_json(task_dir / "ablate_rediscover.json", result)
    _write_report(task_dir / "ablate_rediscover.md", result)
    print(json.dumps({"status": "complete", "task": task, "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()

