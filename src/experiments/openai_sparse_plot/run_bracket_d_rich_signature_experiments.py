from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_d_rich_signatures import (
    SIGNATURE_VARIANTS,
    abstract_signature,
    build_relation_specs_for_split,
    c_signature_from_readout_outputs,
    clean_activation_signature_for_site,
    d_acceptance,
    d_calibration_score,
    depth_target_matrix,
    feature_names_for_signature,
    fit_ridge_readout,
    generate_content_split_multidepth_examples,
    is_readout_valid,
    load_full_localized_candidate_universe,
    quality_to_json,
    r_calibration_score,
    readout_moves_toward_source,
    readout_preserves_base,
    readout_quality,
    relation_counts,
    selected_support_overlap,
    selector_payload_from_signatures,
    split_summary,
    weights_from_ranked,
)
from .bracket_multidepth import DEFAULT_NUMERIC_CONTENTS, MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_d_rich_signatures_20260702")


@dataclass
class CleanRun:
    example_id: str
    token_ids: torch.Tensor
    final_position: int
    margin: float
    predicted_close_count: int
    correct: bool
    features_by_site: dict[str, float]
    feature_vector: tuple[float, ...]
    depth_phi: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-localized-circuit bracket D rich-signature PLOT experiments.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--max-records-per-relation", type=int, default=6)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--readout-valid-threshold", type=float, default=0.90)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    return parser.parse_args()


def _parse_int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("empty integer grid")
    return values


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("empty float grid")
    return values


def _hook_regex(sites: Sequence[ChannelSite]) -> str:
    hooks = sorted({site.hook_key for site in sites})
    return "^(?:" + "|".join(re.escape(hook) for hook in hooks) + ")$"


def _feature_dict(cache: Mapping[str, torch.Tensor], sites: Sequence[ChannelSite], position: int) -> dict[str, float]:
    return {site.site_id: float(cache[site.hook_key][0, int(position), site.channel]) for site in sites}


def _feature_vector(features: Mapping[str, float], sites: Sequence[ChannelSite]) -> tuple[float, ...]:
    return tuple(float(features[site.site_id]) for site in sites)


def _bracket_margin(logits: torch.Tensor, *, single_close_token_id: int, double_close_token_id: int) -> float:
    last = logits[0, -1]
    return float(last[double_close_token_id] - last[single_close_token_id])


def _sign_from_margin(margin: float) -> int:
    return 2 if float(margin) > 0 else 1


def _make_weighted_patch_from_features(
    sites: Sequence[ChannelSite],
    *,
    source_features: Mapping[str, float],
    position: int,
    weights_by_site: Mapping[str, float],
    strength: float,
) -> dict[str, Any]:
    by_hook: dict[str, list[ChannelSite]] = {}
    for site in sites:
        by_hook.setdefault(site.hook_key, []).append(site)
    interventions: dict[str, Any] = {}
    for hook_key, hook_sites in by_hook.items():
        patch_specs = tuple(
            (
                int(site.channel),
                int(position),
                float(source_features[site.site_id]),
                float(weights_by_site.get(site.site_id, 0.0)),
            )
            for site in hook_sites
        )

        def _patch(tensor: torch.Tensor, *, patch_specs: tuple[tuple[int, int, float, float], ...] = patch_specs) -> torch.Tensor:
            patched = tensor.clone()
            alpha = float(strength)
            for channel, pos, source_value, weight in patch_specs:
                base_value = patched[0, pos, channel]
                src = torch.tensor(float(source_value), device=patched.device, dtype=patched.dtype)
                patched[0, pos, channel] = base_value + alpha * float(weight) * (src - base_value)
            return patched

        interventions[hook_key] = _patch
    return interventions


def _collect_clean_runs(
    model: Any,
    examples: Sequence[MultiDepthBracketExample],
    *,
    sites: Sequence[ChannelSite],
    readout: Any | None,
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
) -> dict[str, CleanRun]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    runs: dict[str, CleanRun] = {}
    regex = _hook_regex(sites)
    for idx, ex in enumerate(examples, start=1):
        token_ids = torch.tensor(ex.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        final_position = int(token_ids.shape[1] - 1)
        with torch.no_grad():
            with hook_recorder(regex=regex) as ctx:
                logits, _, _ = model(token_ids)
        cache = {k: v.detach().cpu() for k, v in ctx.items()}
        features = _feature_dict(cache, sites, final_position)
        vector = _feature_vector(features, sites)
        depth_phi = tuple(readout.predict(vector)) if readout is not None else ()
        margin = _bracket_margin(
            logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        predicted = _sign_from_margin(margin)
        runs[ex.example_id] = CleanRun(
            example_id=ex.example_id,
            token_ids=token_ids,
            final_position=final_position,
            margin=margin,
            predicted_close_count=predicted,
            correct=predicted == ex.close_count,
            features_by_site=features,
            feature_vector=vector,
            depth_phi=depth_phi,
        )
        if idx % 32 == 0:
            print(f"collected clean runs {idx}/{len(examples)}", flush=True)
    return runs


def _fit_readout(
    *,
    examples: Sequence[MultiDepthBracketExample],
    clean_runs_without_phi: Mapping[str, CleanRun],
    candidate_sites: Sequence[ChannelSite],
    max_depth: int,
    alpha: float,
) -> Any:
    fit_examples = [ex for ex in examples if ex.split == "Dfit"]
    x = torch.tensor([clean_runs_without_phi[ex.example_id].feature_vector for ex in fit_examples], dtype=torch.float32)
    y = depth_target_matrix(fit_examples, max_depth=max_depth)
    return fit_ridge_readout(x, y, alpha=float(alpha), feature_site_ids=[site.site_id for site in candidate_sites])


def _readout_qualities(
    *,
    readout: Any,
    examples: Sequence[MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    max_depth: int,
) -> dict[str, Any]:
    qualities = {}
    for split in ("Dfit", "Dcal", "Dte"):
        split_examples = [ex for ex in examples if ex.split == split]
        x = torch.tensor([clean_runs[ex.example_id].feature_vector for ex in split_examples], dtype=torch.float32)
        y = depth_target_matrix(split_examples, max_depth=max_depth)
        qualities[split] = readout_quality(readout, split=split, x=x, y=y)
    return qualities


def _run_weighted_patch(
    model: Any,
    *,
    base: CleanRun,
    source: CleanRun,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[float, int, dict[str, float], tuple[float, ...]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = _make_weighted_patch_from_features(
        patch_sites,
        source_features=source.features_by_site,
        position=base.final_position,
        weights_by_site=weights_by_site,
        strength=float(strength),
    )
    with torch.no_grad():
        with hook_recorder(regex=_hook_regex(record_sites), interventions=interventions) as ctx:
            logits, _, _ = model(base.token_ids)
    cache = {k: v.detach().cpu() for k, v in ctx.items()}
    features = _feature_dict(cache, record_sites, base.final_position)
    vector = _feature_vector(features, record_sites)
    margin = _bracket_margin(
        logits.detach().cpu(),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    return margin, _sign_from_margin(margin), features, vector


def _patch_validation_records(
    *,
    model: Any,
    handle_id: str,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: Any,
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = clean_runs[base_ex.example_id]
        source = clean_runs[source_ex.example_id]
        patched_margin, patched_close_count, _features, vector = _run_weighted_patch(
            model,
            base=base,
            source=source,
            patch_sites=patch_sites,
            weights_by_site=weights_by_site,
            strength=float(strength),
            record_sites=record_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        patched_phi = tuple(readout.predict(vector))
        moves = readout_moves_toward_source(base_phi=base.depth_phi, source_phi=source.depth_phi, patched_phi=patched_phi)
        preserves = readout_preserves_base(base_example=base_ex, patched_phi=patched_phi)
        records.append(
            {
                "handle_id": handle_id,
                "site_ids": [site.site_id for site in patch_sites],
                "weights_by_site": dict(weights_by_site),
                "strength": float(strength),
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_example_id": base_ex.example_id,
                "source_example_id": source_ex.example_id,
                "base_depth": base_ex.depth,
                "source_depth": source_ex.depth,
                "base_close_count": base_ex.close_count,
                "source_close_count": source_ex.close_count,
                "base_margin": base.margin,
                "source_margin": source.margin,
                "patched_margin": patched_margin,
                "patched_close_count": patched_close_count,
                "output_preserves_base": patched_close_count == base_ex.close_count,
                "output_matches_source": patched_close_count == source_ex.close_count,
                "output_flips": patched_close_count != base_ex.close_count,
                "base_depth_phi": list(base.depth_phi),
                "source_depth_phi": list(source.depth_phi),
                "patched_depth_phi": list(patched_phi),
                "readout_moves_toward_source": moves,
                "readout_preserves_base": preserves,
            }
        )
    return records


def _clean_summary(examples: Sequence[MultiDepthBracketExample], runs: Mapping[str, CleanRun]) -> dict[str, Any]:
    rows = []
    for ex in examples:
        run = runs[ex.example_id]
        rows.append(
            {
                "example_id": ex.example_id,
                "split": ex.split,
                "depth": ex.depth,
                "close_count": ex.close_count,
                "context_family": ex.context_family,
                "margin": run.margin,
                "predicted_close_count": run.predicted_close_count,
                "correct": run.correct,
            }
        )
    return {
        "n": len(rows),
        "accuracy": sum(1.0 for row in rows if row["correct"]) / len(rows),
        "by_split": {
            split: sum(1.0 for row in rows if row["split"] == split and row["correct"]) / max(1, sum(1 for row in rows if row["split"] == split))
            for split in sorted({row["split"] for row in rows})
        },
    }


def _single_site_patch_signatures(
    *,
    model: Any,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: Any,
    candidate_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    a_by_site: dict[str, list[float]] = {site.site_id: [] for site in candidate_sites}
    c_by_site: dict[str, list[float]] = {site.site_id: [] for site in candidate_sites}
    for idx, site in enumerate(candidate_sites, start=1):
        for spec in specs:
            base = clean_runs[spec.base_id]
            source = clean_runs[spec.source_id]
            margin, _predicted, _features, vector = _run_weighted_patch(
                model,
                base=base,
                source=source,
                patch_sites=(site,),
                weights_by_site={site.site_id: 1.0},
                strength=1.0,
                record_sites=candidate_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            a_by_site[site.site_id].append(float(margin - base.margin))
            patched_phi = readout.predict(vector)
            c_by_site[site.site_id].extend(c_signature_from_readout_outputs(base.depth_phi, patched_phi))
        if idx % 10 == 0 or idx == len(candidate_sites):
            print(f"single-site patch signatures {idx}/{len(candidate_sites)}", flush=True)
    return ({k: tuple(v) for k, v in a_by_site.items()}, {k: tuple(v) for k, v in c_by_site.items()})


def _calibrate_variant(
    *,
    model: Any,
    variant: str,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_lookup: Mapping[str, ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: Any,
    record_sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    rows = []
    is_r_baseline = variant == "A_R_output"
    for k in k_grid:
        weights = weights_from_ranked(ranked_sites, k=int(k))
        patch_sites = tuple(site_lookup[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"{variant}_top{k}_lambda{strength:g}"
            records = _patch_validation_records(
                model=model,
                handle_id=handle_id,
                patch_sites=patch_sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
                examples=examples,
                clean_runs=clean_runs,
                readout=readout,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            from .bracket_d_rich_signatures import summarize_validation_records

            summary = summarize_validation_records(records)
            score = r_calibration_score(records) if is_r_baseline else d_calibration_score(summary)
            rows.append(
                {
                    "handle_id": handle_id,
                    "variant": variant,
                    "k": int(k),
                    "strength": float(strength),
                    "site_ids": [site.site_id for site in patch_sites],
                    "weights_by_site": weights,
                    "summary": summary,
                    "calibration_score": score,
                    "overlap": selected_support_overlap([site.site_id for site in patch_sites]),
                }
            )
        print(f"calibrated {variant} K={k}", flush=True)
    best = sorted(rows, key=lambda row: (-float(row["calibration_score"]), int(row["k"]), abs(float(row["strength"]) - 1.0)))[0]
    return {"grid": rows, "best": best}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Bracket D Rich Signature Experiments",
        "",
        "Candidate universe:",
        "",
        f"- candidate source: `{payload['candidate_manifest']['candidate_source']}`",
        f"- candidate count: `{payload['candidate_manifest']['candidate_count']}`",
        f"- CSV SHA256: `{payload['candidate_manifest']['candidate_csv_sha256']}`",
        f"- no filtering applied: `{payload['candidate_manifest']['no_filtering_applied']}`",
        "",
        "## Clean Behavior And Readout",
        "",
        f"- clean accuracy: `{payload['clean']['accuracy']:.3f}`",
        f"- frozen readout valid: `{payload['frozen_readout']['valid']}`",
    ]
    for split, row in payload["frozen_readout"]["quality"].items():
        lines.append(
            f"- {split} threshold macro accuracy: `{row['threshold_macro_accuracy']:.3f}`, norm-D MAE: `{row['norm_depth_mae']:.3f}`"
        )
    lines.extend(["", "## Signature Results", ""])
    lines.append("| variant | calibrated K | lambda | score | D validated | late/readout overlap | selected sites |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for variant in payload["variant_order"]:
        result = payload["experiments"].get(variant, {})
        if result.get("status") == "skipped":
            lines.append(f"| `{variant}` | - | - | - | `False` | - | skipped: {result.get('reason')} |")
            continue
        held = result["heldout"]
        best = result["calibration"]["best"]
        acceptance = held["acceptance"]
        overlap = best["overlap"]
        lines.append(
            f"| `{variant}` | {best['k']} | {best['strength']:.3f} | {best['calibration_score']:.3f} | "
            f"`{acceptance['D_validated']}` | `{overlap['late_or_readout_overlapping']}` | "
            f"`{', '.join(best['site_ids'])}` |"
        )
    lines.extend(["", "## Conclusion", ""])
    accepted = [variant for variant, result in payload["experiments"].items() if result.get("heldout", {}).get("acceptance", {}).get("D_validated")]
    if accepted:
        lines.append(f"Validated D signatures: `{', '.join(accepted)}`.")
    else:
        lines.append("No variant validated a full `D = active depth` handle under the heldout criteria.")
    lines.append("Experiment A is an output-R baseline and is not interpreted as evidence for D.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "signatures").mkdir(exist_ok=True)
    (args.out_dir / "calibration").mkdir(exist_ok=True)
    (args.out_dir / "heldout").mkdir(exist_ok=True)

    depths = parse_depths(args.depths)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)
    device = "cuda" if args.cuda else "cpu"

    print("loading full localized bracket candidate universe", flush=True)
    universe = load_full_localized_candidate_universe(args.candidate_node_csv, expected_node_count=args.expected_node_count)
    candidate_sites = universe.sites
    site_lookup = {site.site_id: site for site in candidate_sites}
    _write_json(args.out_dir / "candidate_manifest.json", universe.manifest())

    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    examples = generate_content_split_multidepth_examples(enc, depths=depths, numeric_contents=DEFAULT_NUMERIC_CONTENTS)
    lookup = {ex.example_id: ex for ex in examples}
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not bool(args.no_flash),
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    print("collecting clean runs without readout", flush=True)
    clean_without_phi = _collect_clean_runs(
        model,
        examples,
        sites=candidate_sites,
        readout=None,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    readout = _fit_readout(
        examples=examples,
        clean_runs_without_phi=clean_without_phi,
        candidate_sites=candidate_sites,
        max_depth=max(depths),
        alpha=args.ridge_alpha,
    )
    print("collecting clean runs with frozen readout", flush=True)
    clean_runs = _collect_clean_runs(
        model,
        examples,
        sites=candidate_sites,
        readout=readout,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    qualities_obj = _readout_qualities(readout=readout, examples=examples, clean_runs=clean_runs, max_depth=max(depths))
    readout_valid = is_readout_valid(qualities_obj, threshold=args.readout_valid_threshold)
    qualities = {split: quality_to_json(row) for split, row in qualities_obj.items()}

    specs = {
        "Dfit": build_relation_specs_for_split(examples, split="Dfit", max_records_per_relation=args.max_records_per_relation),
        "Dcal": build_relation_specs_for_split(examples, split="Dcal", max_records_per_relation=args.max_records_per_relation),
        "Dte": build_relation_specs_for_split(examples, split="Dte", max_records_per_relation=args.max_records_per_relation),
    }
    print(f"Dfit relation counts: {relation_counts(specs['Dfit'])}", flush=True)
    print(f"Dcal relation counts: {relation_counts(specs['Dcal'])}", flush=True)
    print(f"Dte relation counts: {relation_counts(specs['Dte'])}", flush=True)

    print("building single-site patched signatures for A and C", flush=True)
    a_neural, c_neural = _single_site_patch_signatures(
        model=model,
        specs=specs["Dfit"],
        examples=lookup,
        clean_runs=clean_runs,
        readout=readout,
        candidate_sites=candidate_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )

    selectors: dict[str, Any] = {}
    for variant in SIGNATURE_VARIANTS:
        if variant == "C_D_frozen_readout" and not readout_valid:
            selectors[variant] = {"status": "skipped", "reason": "frozen D readout failed Dcal/Dte quality threshold"}
            _write_json(args.out_dir / "signatures" / "experiment_C_D_frozen_readout.json", {variant: selectors[variant]})
            continue
        abstract = abstract_signature(variant, specs["Dfit"], lookup, max_depth=max(depths))
        if variant == "A_R_output":
            neural_by_site = a_neural
        elif variant.startswith("B_"):
            neural_by_site = {
                site.site_id: clean_activation_signature_for_site(
                    variant,
                    site.site_id,
                    specs["Dfit"],
                    lookup,
                    {ex_id: run.features_by_site for ex_id, run in clean_runs.items()},
                )
                for site in candidate_sites
            }
        elif variant == "C_D_frozen_readout":
            neural_by_site = c_neural
        else:
            raise ValueError(variant)
        selectors[variant] = {
            **selector_payload_from_signatures(
                variant=variant,
                abstract=abstract,
                neural_by_site=neural_by_site,
                epsilon=args.selector_epsilon,
                beta=args.selector_beta,
            ),
            "feature_names": list(feature_names_for_signature(variant, specs["Dfit"])),
        }
        out_name = {
            "A_R_output": "experiment_A_R_output.json",
            "C_D_frozen_readout": "experiment_C_D_frozen_readout.json",
        }.get(variant, "experiment_B_D_clean_activation.json")
        existing = {}
        path = args.out_dir / "signatures" / out_name
        if path.exists() and out_name == "experiment_B_D_clean_activation.json":
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing[variant] = selectors[variant]
        _write_json(path, existing)

    experiments: dict[str, Any] = {}
    for variant in SIGNATURE_VARIANTS:
        selector = selectors[variant]
        if selector.get("status") == "skipped":
            experiments[variant] = selector
            continue
        print(f"calibrating {variant}", flush=True)
        ranked = selector["selectors"]["raw_cosine_uot"]["ranked_sites"]
        calibration = _calibrate_variant(
            model=model,
            variant=variant,
            ranked_sites=ranked,
            site_lookup=site_lookup,
            specs=specs["Dcal"],
            examples=lookup,
            clean_runs=clean_runs,
            readout=readout,
            record_sites=candidate_sites,
            k_grid=k_grid,
            strength_grid=strength_grid,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        _write_jsonl(args.out_dir / "calibration" / f"{variant}.jsonl", calibration["grid"])
        best = calibration["best"]
        patch_sites = tuple(site_lookup[site_id] for site_id in best["weights_by_site"])
        print(f"heldout {variant}", flush=True)
        from .bracket_d_rich_signatures import summarize_validation_records

        heldout_records = _patch_validation_records(
            model=model,
            handle_id=str(best["handle_id"]),
            patch_sites=patch_sites,
            weights_by_site=best["weights_by_site"],
            strength=float(best["strength"]),
            specs=specs["Dte"],
            examples=lookup,
            clean_runs=clean_runs,
            readout=readout,
            record_sites=candidate_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        heldout_summary = summarize_validation_records(heldout_records)
        heldout = {
            "records": heldout_records,
            "summary": heldout_summary,
            "acceptance": d_acceptance(heldout_summary),
        }
        _write_jsonl(args.out_dir / "heldout" / f"{variant}.jsonl", heldout_records)
        experiments[variant] = {
            "selector": selector,
            "calibration": calibration,
            "heldout": heldout,
        }

    payload = {
        "model": args.model,
        "model_info": model_info,
        "candidate_manifest": universe.manifest(),
        "depths": list(depths),
        "splits": split_summary(examples),
        "max_records_per_relation": int(args.max_records_per_relation),
        "relation_counts": {split: relation_counts(rows) for split, rows in specs.items()},
        "clean": _clean_summary(examples, clean_runs),
        "frozen_readout": {
            "valid": readout_valid,
            "valid_threshold": float(args.readout_valid_threshold),
            "quality": qualities,
            "readout": readout.to_json(),
            "fit_split": "Dfit",
            "used_for_matching_variants": ["C_D_frozen_readout"] if readout_valid else [],
            "used_for_heldout_validation": list(SIGNATURE_VARIANTS),
        },
        "variant_order": list(SIGNATURE_VARIANTS),
        "experiments": experiments,
    }
    _write_json(args.out_dir / "bracket_d_rich_signature_experiments.json", payload)
    _write_report(args.out_dir / "bracket_d_rich_signature_experiments.md", payload)
    print(json.dumps({"out_dir": str(args.out_dir), "readout_valid": readout_valid}, indent=2), flush=True)


if __name__ == "__main__":
    main()
