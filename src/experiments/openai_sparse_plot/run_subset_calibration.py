from __future__ import annotations

import argparse
import json
from itertools import combinations
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
    parser.add_argument(
        "--intact-model",
        action="store_true",
        help="Run one coarse localization round on the intact model over the full candidate set.",
    )
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _token_ids(task: str, enc: Any) -> tuple[int, int]:
    if task == "quote":
        tokens = quote_token_ids(enc)
        return int(tokens["single"]), int(tokens["double"])
    return int(enc.encode("]\n")[0]), int(enc.encode("]]\n")[0])


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    coarse = payload.get("experiment") == "coarse_localization"
    title = "Coarse Localization" if coarse else "Ablate And Rediscover"
    lines = [f"# {title}: {payload['task'].title()}", ""]
    lines.append(f"- full localized candidate universe: `{payload['candidate_count']}`")
    lines.append(
        "- candidate filtering: none; the model is intact and all candidates are searched"
        if coarse
        else "- candidate filtering: only sites explicitly disabled by preceding rounds are removed from the primary search"
    )
    lines.extend(
        [
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
                f"- {'intact' if coarse else 'clamped'} Dte clean accuracy: `{row['clean_accuracy']['Dte']:.3f}`",
                f"- selected handle: `{', '.join(row['selected_site_ids'])}`",
                f"- Dcal score: `{row['calibration_best']['summary']['score']:.3f}`",
                f"- Dte rates: `{json.dumps(row['heldout']['rates'], sort_keys=True)}`",
                f"- diagnostic behavioral pass: `{row['diagnostic_handle_pass']}`",
                *([] if coarse else [f"- certified natural redundancy: `{row['redundancy_certified']}`"]),
                "",
            ]
        )
    lines.extend(["## Conclusion", "", f"{payload['conclusion']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    task = str(args.task)
    initial_disabled = () if args.intact_model else INITIAL_DISABLED[task]
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
    if initial_disabled:
        means_path = args.necessity_root / task / "Dfit_task_means.pt"
        if not means_path.exists():
            raise FileNotFoundError(f"missing frozen Dfit task means: {means_path}")
        hook_means = torch.load(means_path, map_location="cpu", weights_only=True)
    else:
        hook_means = {}
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
            "experiment": "coarse_localization" if args.intact_model else "ablate_and_rediscover",
            "task": task,
            "model": model_name,
            "candidate_count": len(circuit.sites),
            "candidate_csv": circuit.csv_path,
            "candidate_csv_sha256": circuit.csv_sha256,
            "candidate_filtering": (
                "none; intact full candidate universe"
                if args.intact_model
                else "none, except explicitly disabled handles in each primary round"
            ),
            "initial_disabled": list(initial_disabled),
            "mean_ablation": (
                None
                if args.intact_model
                else "unconditional all-token Dfit task mean frozen by the necessity audit"
            ),
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
    disabled_ids = list(initial_disabled)
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
        # abstract = abstract_signature(pairs_by_split["Dfit"], examples_by_id)
        # selector = match_signatures(
        #     abstract,
        #     neural_by_site,
        #     epsilon=args.selector_epsilon,
        #     beta=args.selector_beta,
        # )

        abstract = abstract_signature(pairs_by_split["Dfit"], examples_by_id)

        if task == "bracket":
            abstract, neural_by_site = normalize_relation_blocks(
                abstract,
                neural_by_site,
                pairs_by_split["Dfit"],
            )

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
        support = selector["ranked"][:8]
        for index in range(len(support) - 1):
            if float(support[index + 1]["weight"]) <= 0.5 * float(support[index]["weight"]):
                support = support[: index + 1]
                break
        calibration_configs: list[HandleConfiguration] = []
        calibration_meta: list[dict[str, Any]] = []
        for size in (1, 2):
            for index, combo in enumerate(combinations(support, size), start=1):
                total = sum(float(row["weight"]) for row in combo)
                weights = {str(row["site_id"]): float(row["weight"]) / total for row in combo}
                for strength in strength_grid:
                    handle_id = f"S{size}_{index}_lambda{strength:g}"
                    calibration_configs.append(HandleConfiguration(handle_id, weights, float(strength)))
                    calibration_meta.append({"handle_id": handle_id, "k": size, "strength": float(strength), "weights": weights})
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
        best_score = max(float(row["summary"]["score"]) for row in calibration_rows)
        best_tier = [dict(row) for row in calibration_rows if abs(float(row["summary"]["score"]) - best_score) < 1e-12]
        best = dict(select_calibration_row(best_tier))
        selected = HandleConfiguration("selected", {str(key): float(value) for key, value in best["weights"].items()}, float(best["strength"]))
        selected_tier = tuple(HandleConfiguration(row["handle_id"], {str(key): float(value) for key, value in row["weights"].items()}, float(row["strength"])) for row in best_tier)
        heldout_margins = evaluate_configurations(
            model,
            selected_tier,
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
        )
        heldout_tier = []
        for row, margins in zip(best_tier, heldout_margins):
            summary = relation_summary(pairs_by_split["Dte"], examples_by_id, margins)
            heldout_tier.append({**row, "heldout": summary, "diagnostic_handle_pass": bool(summary["all_rates_at_least_0_90"]), "redundancy_certified": can_certify_redundancy(clean_dte_accuracy=clean["Dte"], heldout_summary=summary)})
        heldout = next(row["heldout"] for row in heldout_tier if row["handle_id"] == best["handle_id"])
        diagnostic_pass = any(row["diagnostic_handle_pass"] for row in heldout_tier)
        redundancy = any(row["redundancy_certified"] for row in heldout_tier)
        round_payload = {
            "round": round_index,
            "disabled_before": list(disabled_ids),
            "candidate_count": len(candidate_sites),
            "clean_accuracy": clean,
            "calibration_best": best,
            "calibration_best_tier": best_tier,
            "effective_support": support,
            "selected_site_ids": list(selected.weights_by_site),
            "heldout": heldout,
            "heldout_tier": heldout_tier,
            "diagnostic_handle_pass": diagnostic_pass,
            "redundancy_certified": redundancy,
        }
        rounds.append(round_payload)
        atomic_json(round_dir / "calibration.json", {"grid": calibration_rows, "best": best, "best_tier": best_tier, "effective_support": support})
        atomic_json(round_dir / "heldout.json", round_payload)
        print(json.dumps(round_payload, indent=2), flush=True)
        if args.intact_model:
            break
        if not redundancy:
            break
        disabled_ids.extend(site_id for site_id in selected.weights_by_site if site_id not in disabled_ids)
    if args.intact_model:
        conclusion = "Coarse causal localization completed on the intact model."
    elif rounds and rounds[-1]["redundancy_certified"]:
        conclusion = "At least one naturally redundant alternative handle was certified after ablation."
    elif rounds and rounds[-1]["diagnostic_handle_pass"]:
        conclusion = (
            "A diagnostic intervention handle was found, but natural redundancy was not certified because the "
            "clamped model did not retain at least 0.90 clean accuracy."
        )
    else:
        conclusion = "No behaviorally valid alternative handle was certified after ablating the learned handle."
    result = {
        "experiment": "coarse_localization" if args.intact_model else "ablate_and_rediscover",
        "task": task,
        "candidate_count": len(circuit.sites),
        "rounds": rounds,
        "conclusion": conclusion,
    }
    output_stem = "coarse_localization" if args.intact_model else "ablate_rediscover"
    atomic_json(task_dir / f"{output_stem}.json", result)
    _write_report(task_dir / f"{output_stem}.md", result)
    print(json.dumps({"status": "complete", "task": task, "conclusion": conclusion}, indent=2))


def normalize_relation_blocks(abstract, neural_by_site, pairs):
    abstract = torch.tensor(abstract, dtype=torch.float32)
    site_ids = list(neural_by_site)

    neural = torch.tensor(
        [neural_by_site[site_id] for site_id in site_ids],
        dtype=torch.float32,
    )

    relations = sorted({pair.relation for pair in pairs})

    for relation in relations:
        indices = [
            i for i, pair in enumerate(pairs)
            if pair.relation == relation
        ]

        scale = max(
            float(torch.sqrt((abstract[indices] ** 2).mean())),
            float(torch.sqrt((neural[:, indices] ** 2).mean())),
            1e-6,
        )

        abstract[indices] /= scale
        neural[:, indices] /= scale

    normalized_neural = {
        site_id: tuple(neural[i].tolist())
        for i, site_id in enumerate(site_ids)
    }

    return tuple(abstract.tolist()), normalized_neural


if __name__ == "__main__":
    main()