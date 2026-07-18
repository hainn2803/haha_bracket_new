from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .handle_necessity import (
    AblationConfiguration,
    NecessityExample,
    atomic_json,
    bank_manifest,
    build_bracket_necessity_bank,
    build_quote_necessity_bank,
    chunked,
    estimate_hook_means,
    evaluate_ablation_configs,
    load_candidate_circuit,
    load_npz,
    pruning_hook_keys,
    save_npz,
    singleton_ranks,
    summarize_ablation_results,
    validate_means,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


DEFAULT_QUOTE_CSV = Path("eval/openai_sparse_plot/string_closing_prune_v2_64/string_closing_circuit_nodes.csv")
DEFAULT_BRACKET_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/frozen_handle_necessity_20260715")


FROZEN_HANDLES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "quote": (
        ("quote_U", ("0.mlp.resid_delta:460",)),
    ),
    "bracket": (
        ("R_mid", ("4.attn.resid_delta:1079",)),
        ("D_published_comparator", ("2.attn.resid_delta:1249",)),
        (
            "R_coarse",
            ("final_resid:1079", "7.mlp.post_act:4133", "7.mlp.resid_delta:2041"),
        ),
        ("R_late_single", ("7.mlp.act_in:1079",)),
        ("R_late_pair", ("7.mlp.act_in:1079", "4.attn.q:1292")),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-handle task-mean necessity audit.")
    parser.add_argument("--task", choices=("quote", "bracket"), required=True)
    parser.add_argument("--circuit-home", type=Path, default=Path(".external/circuit_sparsity"))
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fit-contents", type=int, default=48)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--content-offset", type=int, default=12000)
    parser.add_argument("--mean-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--config-chunk-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dense-kernels", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _configurations(task: str, node_ids: Sequence[str]) -> tuple[AblationConfiguration, ...]:
    configs = [
        AblationConfiguration(
            config_id=f"single::{node_id}",
            site_ids=(node_id,),
            scope="global",
            role="all-retained-singleton global task-mean ablation",
        )
        for node_id in node_ids
    ]
    for handle_id, site_ids in FROZEN_HANDLES[task]:
        configs.append(
            AblationConfiguration(
                config_id=f"handle::{handle_id}::global",
                site_ids=site_ids,
                scope="global",
                role="frozen learned/published handle set-level necessity",
            )
        )
        configs.append(
            AblationConfiguration(
                config_id=f"handle::{handle_id}::position",
                site_ids=site_ids,
                scope="handle_position",
                role="same position used by the executable causal handle",
            )
        )
        if len(site_ids) > 1:
            for omitted in site_ids:
                kept = tuple(value for value in site_ids if value != omitted)
                configs.append(
                    AblationConfiguration(
                        config_id=f"leave_one_out::{handle_id}::omit::{omitted}",
                        site_ids=kept,
                        scope="global",
                        role="predeclared leave-one-out set ablation",
                    )
                )
    configs.append(
        AblationConfiguration(
            config_id="all_retained_nodes",
            site_ids=tuple(node_ids),
            scope="global",
            role="entire released localized circuit mean-ablated",
        )
    )
    return tuple(configs)


def _manifest(
    args: argparse.Namespace,
    *,
    task_dir: Path,
    model_name: str,
    candidate_count: int,
    candidate_sha256: str,
    hook_keys: Sequence[str],
    configs: Sequence[AblationConfiguration],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "frozen_handle_necessity_audit",
        "task": args.task,
        "task_dir": str(task_dir),
        "model": model_name,
        "candidate_count": int(candidate_count),
        "candidate_csv_sha256": candidate_sha256,
        "candidate_filtering": "none: every retained singleton is globally ablated",
        "mean_ablation": {
            "definition": "unconditional empirical mean over every token in the sealed Dfit task bank",
            "fit_split_only": True,
            "exact_OpenAI_pretraining_mean_replication": False,
            "reason": "released artifacts do not contain pretraining activation means",
        },
        "settings": ["full_sparse_model", "reconstructed_circuit_only"],
        "circuit_only_definition": "all non-retained coordinates at every OpenAI pruning hook are set to Dfit task means",
        "circuit_only_ablation_gate": {
            "metric": "clean contrast accuracy",
            "minimum": 0.90,
            "reason": "necessity effects are not interpretable against a task-incompetent reconstructed baseline",
        },
        "hook_keys": list(hook_keys),
        "configuration_count": len(configs),
        "configurations": [
            {
                "config_id": row.config_id,
                "site_ids": list(row.site_ids),
                "scope": row.scope,
                "role": row.role,
            }
            for row in configs
        ],
        "selection_or_recalibration": False,
        "Dte_used_for": "final necessity evaluation and confidence intervals only",
        "bank": bank,
        "bootstrap_unit": "content_id",
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed),
    }


def _load_or_estimate_means(
    model: Any,
    fit_examples: Sequence[NecessityExample],
    *,
    path: Path,
    counts_path: Path,
    hook_keys: Sequence[str],
    device: str,
    batch_size: int,
    resume: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if resume and path.exists() and counts_path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        return {str(key): value.to(torch.float32) for key, value in payload.items()}, {
            str(key): int(value) for key, value in counts.items()
        }
    means, counts = estimate_hook_means(
        model,
        fit_examples,
        hook_keys=hook_keys,
        device=device,
        batch_size=int(batch_size),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(means, path)
    atomic_json(counts_path, counts)
    return means, counts


def _evaluate_setting(
    model: Any,
    examples: Sequence[NecessityExample],
    *,
    setting: str,
    configs: Sequence[AblationConfiguration],
    sites: Sequence[Any],
    means: Mapping[str, torch.Tensor],
    hook_keys: Sequence[str],
    retained_by_hook: Mapping[str, Sequence[int]],
    task_dir: Path,
    device: str,
    max_batch_size: int,
    config_chunk_size: int,
    resume: bool,
) -> tuple[dict[str, np.ndarray], list[AblationConfiguration]]:
    circuit_only = setting == "circuit_only"
    clean_config = AblationConfiguration(config_id="clean", site_ids=(), scope="none", role="baseline")
    clean_path = task_dir / "evaluations" / f"{setting}_clean.npz"
    if resume and clean_path.exists():
        clean = load_npz(clean_path)
    else:
        clean = evaluate_ablation_configs(
            model,
            examples,
            configs=(clean_config,),
            sites=sites,
            hook_means=means,
            hook_keys=hook_keys,
            retained_by_hook=retained_by_hook,
            circuit_only=circuit_only,
            device=device,
            max_batch_size=int(max_batch_size),
        )
        save_npz(clean_path, clean)
    all_payload = {key: [] for key in clean}
    ordered_configs: list[AblationConfiguration] = []
    for start, rows in chunked(configs, int(config_chunk_size)):
        chunk_path = task_dir / "evaluations" / f"{setting}_configs_{start:04d}.npz"
        if resume and chunk_path.exists():
            payload = load_npz(chunk_path)
        else:
            payload = evaluate_ablation_configs(
                model,
                examples,
                configs=rows,
                sites=sites,
                hook_means=means,
                hook_keys=hook_keys,
                retained_by_hook=retained_by_hook,
                circuit_only=circuit_only,
                device=device,
                max_batch_size=int(max_batch_size),
            )
            save_npz(chunk_path, payload)
        for key, value in payload.items():
            all_payload[key].append(value)
        ordered_configs.extend(rows)
        print(f"{args.task} {setting}: configurations {start + len(rows)}/{len(configs)}", flush=True)
    combined = {key: np.concatenate(values, axis=0) for key, values in all_payload.items()}
    return {**combined, "_clean_nll": clean["nll"], "_clean_margin": clean["margin"], "_clean_correct": clean["correct"]}, ordered_configs


def _evaluate_clean_setting(
    model: Any,
    examples: Sequence[NecessityExample],
    *,
    setting: str,
    sites: Sequence[Any],
    means: Mapping[str, torch.Tensor],
    hook_keys: Sequence[str],
    retained_by_hook: Mapping[str, Sequence[int]],
    task_dir: Path,
    device: str,
    max_batch_size: int,
    resume: bool,
) -> dict[str, np.ndarray]:
    path = task_dir / "evaluations" / f"{setting}_clean.npz"
    if resume and path.exists():
        return load_npz(path)
    clean = evaluate_ablation_configs(
        model,
        examples,
        configs=(AblationConfiguration("clean", (), "none", "baseline"),),
        sites=sites,
        hook_means=means,
        hook_keys=hook_keys,
        retained_by_hook=retained_by_hook,
        circuit_only=setting == "circuit_only",
        device=device,
        max_batch_size=int(max_batch_size),
    )
    save_npz(path, clean)
    return clean


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [f"# Frozen-Handle Necessity Audit: {payload['task'].title()}", ""]
    lines.append(f"- candidate singletons: `{payload['candidate_count']}`")
    lines.append("- candidate filtering: `none`")
    lines.append("- mean: unconditional Dfit task-distribution mean over all tokens")
    lines.append("- exact OpenAI pretraining-mean replication: `False`")
    lines.append("")
    lines.append("## Circuit Sufficiency")
    lines.append("")
    sufficiency = payload["circuit_sufficiency"]
    lines.append(f"- full-model clean accuracy: `{sufficiency['full_accuracy']:.3f}`")
    lines.append(f"- circuit-only clean accuracy: `{sufficiency['circuit_accuracy']:.3f}`")
    lines.append(f"- circuit-only NLL increase: `{sufficiency['nll_increase']:.6f}`")
    lines.append("")
    lines.append("## Frozen Handles")
    lines.append("")
    lines.append(
        "| setting | handle | scope | contrast-loss delta | margin drop | accuracy after | "
        "ranks: accuracy / margin / contrast / NLL |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in payload["frozen_handle_rows"]:
        ranks = row.get("singleton_ranks", {})
        lines.append(
            f"| `{row['setting']}` | `{row['config_id']}` | `{row['scope']}` | "
            f"{row['mean_contrast_loss_delta']:.6f} | {row['mean_margin_drop']:.6f} | "
            f"{row['ablated_contrast_accuracy']:.3f} | "
            f"`{ranks.get('accuracy')} / {ranks.get('margin')} / "
            f"{ranks.get('contrast_loss')} / {ranks.get('nll')}` |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for line in payload["conclusions"]:
        lines.append(f"- {line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global args
    args = parse_args()
    resume = not bool(args.no_resume)
    task = str(args.task)
    model_name = "csp_yolo1" if task == "quote" else "csp_yolo2"
    expected_count = 64 if task == "quote" else 133
    candidate_csv = args.candidate_csv or (DEFAULT_QUOTE_CSV if task == "quote" else DEFAULT_BRACKET_CSV)
    task_dir = args.out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_candidate_circuit(candidate_csv, expected_count=expected_count)
    encoding = make_tinypython_encoding(args.circuit_home)
    examples = (
        build_quote_necessity_bank(
            encoding,
            fit_contents=int(args.fit_contents),
            test_contents=int(args.test_contents),
            content_offset=int(args.content_offset),
        )
        if task == "quote"
        else build_bracket_necessity_bank(
            encoding,
            fit_contents=int(args.fit_contents),
            test_contents=int(args.test_contents),
            content_offset=int(args.content_offset),
        )
    )
    manifest_bank = bank_manifest(examples)
    if not manifest_bank["content_splits_disjoint"]:
        raise ValueError("Dfit and Dte content banks overlap")
    fit_examples = tuple(row for row in examples if row.split == "Dfit")
    test_examples = tuple(row for row in examples if row.split == "Dte")
    model, model_info = load_sparse_gpt_model(
        model_name=model_name,
        circuit_home=args.circuit_home,
        cuda=bool(args.cuda),
        flash=True,
        grad_checkpointing=False,
    )
    sparse_records = () if args.dense_kernels else convert_transformer_linears_to_sparse(model)
    model_info["execution_linear_kernel"] = "dense" if args.dense_kernels else "exact_sparse_csr"
    model_info["sparse_conversion"] = [row.to_json() for row in sparse_records]
    atomic_json(task_dir / "model_info.json", model_info)
    n_layer = int(model_info["filtered_config"]["n_layer"])
    hook_keys = pruning_hook_keys(n_layer)
    unknown_hooks = set(circuit.retained_by_hook) - set(hook_keys)
    if unknown_hooks:
        raise ValueError(f"candidate hooks are outside the pruning hook universe: {sorted(unknown_hooks)}")
    configs = _configurations(task, circuit.node_ids)
    manifest = _manifest(
        args,
        task_dir=task_dir,
        model_name=model_name,
        candidate_count=len(circuit.sites),
        candidate_sha256=circuit.csv_sha256,
        hook_keys=hook_keys,
        configs=configs,
        bank=manifest_bank,
    )
    manifest_path = task_dir / "run_manifest.json"
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            legacy = dict(manifest)
            legacy.pop("circuit_only_ablation_gate", None)
            if existing != legacy:
                raise ValueError("run manifest differs from existing checkpoint")
            atomic_json(manifest_path, manifest)
    else:
        atomic_json(manifest_path, manifest)
    device = "cuda" if args.cuda else "cpu"
    means, counts = _load_or_estimate_means(
        model,
        fit_examples,
        path=task_dir / "Dfit_task_means.pt",
        counts_path=task_dir / "Dfit_task_mean_counts.json",
        hook_keys=hook_keys,
        device=device,
        batch_size=int(args.mean_batch_size),
        resume=resume,
    )
    validate_means(means, hook_keys=hook_keys, sites=circuit.sites)
    atomic_json(
        task_dir / "mean_manifest.json",
        {
            "split": "Dfit",
            "fit_example_count": len(fit_examples),
            "hook_count": len(means),
            "counts": counts,
            "Dte_used": False,
        },
    )
    summaries_by_setting: dict[str, list[dict[str, Any]]] = {}
    clean_by_setting: dict[str, dict[str, np.ndarray]] = {}
    for setting in ("full_model", "circuit_only"):
        if setting == "circuit_only":
            clean = _evaluate_clean_setting(
                model,
                test_examples,
                setting=setting,
                sites=circuit.sites,
                means=means,
                hook_keys=hook_keys,
                retained_by_hook=circuit.retained_by_hook,
                task_dir=task_dir,
                device=device,
                max_batch_size=int(args.max_batch_size),
                resume=resume,
            )
            clean_by_setting[setting] = clean
            clean_accuracy = float(np.asarray(clean["correct"]).mean())
            if clean_accuracy < float(manifest["circuit_only_ablation_gate"]["minimum"]):
                summaries_by_setting[setting] = []
                atomic_json(
                    task_dir / "circuit_only_status.json",
                    {
                        "status": "invalid_baseline_ablation_skipped",
                        "clean_contrast_accuracy": clean_accuracy,
                        "minimum_required": manifest["circuit_only_ablation_gate"]["minimum"],
                        "reason": manifest["circuit_only_ablation_gate"]["reason"],
                    },
                )
                continue
        payload, ordered = _evaluate_setting(
            model,
            test_examples,
            setting=setting,
            configs=configs,
            sites=circuit.sites,
            means=means,
            hook_keys=hook_keys,
            retained_by_hook=circuit.retained_by_hook,
            task_dir=task_dir,
            device=device,
            max_batch_size=int(args.max_batch_size),
            config_chunk_size=int(args.config_chunk_size),
            resume=resume,
        )
        clean = {
            "nll": payload.pop("_clean_nll"),
            "margin": payload.pop("_clean_margin"),
            "correct": payload.pop("_clean_correct"),
        }
        clean_by_setting[setting] = clean
        summaries = summarize_ablation_results(
            test_examples,
            ordered,
            payload,
            clean,
            bootstrap_repetitions=int(args.bootstrap_repetitions),
            bootstrap_seed=int(args.bootstrap_seed) + (0 if setting == "full_model" else 100000),
        )
        summaries_by_setting[setting] = summaries
        atomic_json(task_dir / f"{setting}_summaries.json", summaries)
    full_clean = clean_by_setting["full_model"]
    circuit_clean = clean_by_setting["circuit_only"]
    if float(np.asarray(full_clean["correct"]).mean()) < 1.0:
        raise RuntimeError("full-model clean Dte contrast accuracy is below 1.0")
    rank_metrics = {
        "accuracy": "mean_accuracy_drop",
        "margin": "mean_margin_drop",
        "contrast_loss": "mean_contrast_loss_delta",
        "nll": "mean_nll_delta",
    }
    ranks = {
        setting: {
            name: singleton_ranks(rows, metric=metric)
            for name, metric in rank_metrics.items()
        }
        for setting, rows in summaries_by_setting.items()
    }
    frozen_rows = []
    frozen_prefixes = ("handle::", "leave_one_out::", "all_retained_nodes")
    for setting, rows in summaries_by_setting.items():
        for row in rows:
            if str(row["config_id"]).startswith(frozen_prefixes) or str(row["config_id"]) == "all_retained_nodes":
                record = {**row, "setting": setting}
                if len(row["site_ids"]) == 1:
                    site_id = str(row["site_ids"][0])
                    record["singleton_ranks"] = {
                        name: rank_map.get(site_id) for name, rank_map in ranks[setting].items()
                    }
                else:
                    record["singleton_ranks"] = {}
                frozen_rows.append(record)
    conclusions = []
    for handle_id, site_ids in FROZEN_HANDLES[task]:
        for setting in ("full_model", "circuit_only"):
            row = next(
                (
                    value
                    for value in frozen_rows
                    if value["setting"] == setting and value["config_id"] == f"handle::{handle_id}::global"
                ),
                None,
            )
            if row is None:
                continue
            conclusions.append(
                f"{handle_id} in {setting}: contrast-loss delta "
                f"{row['mean_contrast_loss_delta']:.6f}, margin drop {row['mean_margin_drop']:.6f}, "
                f"accuracy {row['ablated_contrast_accuracy']:.3f}."
            )
    if not summaries_by_setting["circuit_only"]:
        conclusions.append(
            "The reconstructed circuit-only baseline failed the preregistered 0.90 clean-accuracy gate; "
            "its handle ablations are not interpreted."
        )
    result = {
        "task": task,
        "candidate_count": len(circuit.sites),
        "candidate_csv_sha256": circuit.csv_sha256,
        "mean_definition": manifest["mean_ablation"],
        "circuit_sufficiency": {
            "full_accuracy": float(np.asarray(full_clean["correct"]).mean()),
            "circuit_accuracy": float(np.asarray(circuit_clean["correct"]).mean()),
            "full_mean_nll": float(np.asarray(full_clean["nll"]).mean()),
            "circuit_mean_nll": float(np.asarray(circuit_clean["nll"]).mean()),
            "nll_increase": float(np.asarray(circuit_clean["nll"]).mean() - np.asarray(full_clean["nll"]).mean()),
        },
        "singleton_ranks": ranks,
        "frozen_handle_rows": frozen_rows,
        "circuit_only_ablation_status": (
            "evaluated" if summaries_by_setting["circuit_only"] else "invalid_baseline_ablation_skipped"
        ),
        "conclusions": conclusions,
    }
    atomic_json(task_dir / "frozen_handle_necessity_audit.json", result)
    _write_report(task_dir / "frozen_handle_necessity_audit.md", result)
    print(json.dumps({"task": task, "status": "complete", "circuit_sufficiency": result["circuit_sufficiency"]}, indent=2))


if __name__ == "__main__":
    main()
