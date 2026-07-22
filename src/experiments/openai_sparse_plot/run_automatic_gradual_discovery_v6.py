"""Blind automatic discovery of the gradual circuit X -> D -> R -> Y.

The code has two clearly separated parts.

Selection (Dfit and Dcal only):
1. Find a late R handle from all candidate sites.
2. Refine R by looking for an upstream handle that controls the frozen R.
3. Find a late D handle that controls the final R.
4. Refine D by looking for an upstream handle that controls the frozen D.

Heldout certification (Dte only):
5. Freeze every selected handle and hyperparameter.
6. Load Dte and certify the directed edges of the final chain.

Dte activations are not created during selection. A runtime lock also prevents
an accidental Dte evaluation before step 6.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ablate_rediscover import (
    ClampedRun,
    HandleConfiguration,
    RediscoveryExample,
    RediscoveryPair,
    abstract_signature,
    atomic_json,
    bank_manifest,
    build_bracket_pairs,
    build_bracket_rediscovery_bank,
    clean_accuracy,
    collect_clamped_runs,
    evaluate_configurations,
    load_candidate_circuit,
    match_signatures,
    relation_summary,
    select_calibration_row,
)
from .activation import ChannelSite
from .bracket_progressive_model_discovery import layer_order
from .graded_evidence import (
    abstract_e_signature as abstract_d_signature,
    build_graded_evidence_bank as build_graded_d_bank,
    build_graded_pairs,
    decoder_metrics,
    e_value as d_value,
    fit_affine_decoder,
    graded_validation_summary,
)
from .progressive_rearly import (
    evaluate_progressive_configurations,
    fit_binary_scalar_readout,
    mediation_summary,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import convert_transformer_linears_to_sparse


FIT_SPLIT = "Dfit"
CAL_SPLIT = "Dcal"
TEST_SPLIT = "Dte"

SELECTION_SPLITS = (FIT_SPLIT, CAL_SPLIT)
ALL_SPLITS = (FIT_SPLIT, CAL_SPLIT, TEST_SPLIT)

PASS_THRESHOLD = 0.90
D_DEFINITION = "active_depth"

JsonDict = dict[str, Any]
Weights = Mapping[str, float]


@dataclass
class Dataset:
    """Examples, intervention pairs, and cached model activations for one bank."""

    examples: tuple[RediscoveryExample, ...]
    pairs: dict[str, tuple[RediscoveryPair, ...]]
    examples_by_id: dict[str, RediscoveryExample]
    runs: dict[str, ClampedRun]


@dataclass
class ExperimentContext:
    """Objects shared by every phase of the experiment."""

    args: argparse.Namespace
    model: Any
    sites: tuple[ChannelSite, ...]
    sites_by_id: dict[str, ChannelSite]
    negative_token_id: int
    positive_token_id: int
    device: str
    heldout_unlocked: bool = False


@dataclass
class HandleCandidate:
    """A calibrated handle plus the readouts used to classify it as D or R."""

    calibration: JsonDict
    weights: dict[str, float]
    r_readout: Any
    r_accuracy: dict[str, float]
    d_decoder: Any
    d_metrics: dict[str, dict[str, float]]
    is_d: bool


@dataclass
class RDiscovery:
    coarse_result: JsonDict
    late: HandleCandidate
    early: HandleCandidate | None
    refinement: JsonDict | None
    chain: list[JsonDict]

    @property
    def final(self) -> HandleCandidate:
        """The most upstream selected R handle."""

        return self.early or self.late


@dataclass
class DDiscovery:
    discovery_result: JsonDict
    direct_validation: JsonDict
    late: HandleCandidate
    early: HandleCandidate | None
    refinement: JsonDict | None
    chain: list[JsonDict]

    @property
    def final(self) -> HandleCandidate:
        """The most upstream selected D handle."""

        return self.early or self.late


# ---------------------------------------------------------------------------
# Setup and data
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic gradual PLOT for X -> D -> R -> Y."
    )
    parser.add_argument(
        "--circuit-home",
        type=Path,
        default=Path(".external/circuit_sparsity"),
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("data/bracket_circuit_nodes.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/automatic_gradual_discovery"),
    )
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--mass-fraction", type=float, default=0.0)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--graded-threshold", type=float, default=0.9)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def parse_strengths(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value.strip())


def setup_experiment(
    args: argparse.Namespace,
) -> tuple[ExperimentContext, Any, JsonDict, list[Any]]:
    """Load the model and the fixed set of 133 candidate sites."""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder = make_tinypython_encoding(args.circuit_home)
    circuit = load_candidate_circuit(args.candidate_csv, expected_count=133)

    model, model_info = load_sparse_gpt_model(
        model_name="csp_yolo2",
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=True,
        grad_checkpointing=False,
    )
    sparse_records = convert_transformer_linears_to_sparse(model)

    context = ExperimentContext(
        args=args,
        model=model,
        sites=circuit.sites,
        sites_by_id={site.site_id: site for site in circuit.sites},
        negative_token_id=int(encoder.encode("]\n")[0]),
        positive_token_id=int(encoder.encode("]]\n")[0]),
        device="cuda" if args.cuda else "cpu",
    )
    return context, encoder, model_info, sparse_records


def collect_runs(
    context: ExperimentContext,
    examples: Sequence[RediscoveryExample],
) -> dict[str, ClampedRun]:
    runs = collect_clamped_runs(
        context.model,
        examples,
        candidate_sites=context.sites,
        disabled_sites=(),
        hook_means={},
        negative_token_id=context.negative_token_id,
        positive_token_id=context.positive_token_id,
        device=context.device,
        max_batch_size=context.args.max_batch_size,
    )
    return dict(runs)


def make_dataset(
    context: ExperimentContext,
    examples: tuple[RediscoveryExample, ...],
    pairs: dict[str, tuple[RediscoveryPair, ...]],
) -> Dataset:
    """Create only Dfit/Dcal activations. Dte remains untouched."""

    selection_examples = tuple(
        example for example in examples if example.split in SELECTION_SPLITS
    )
    return Dataset(
        examples=examples,
        pairs=pairs,
        examples_by_id={example.example_id: example for example in examples},
        runs=collect_runs(context, selection_examples),
    )


def build_coarse_dataset(context: ExperimentContext, encoder: Any) -> Dataset:
    examples = build_bracket_rediscovery_bank(
        encoder,
        fit_contents=48,
        cal_contents=24,
        test_contents=24,
        content_offset=13000,
    )
    pairs = {
        split: build_bracket_pairs(
            examples,
            split=split,
            records_per_relation=100,
        )
        for split in ALL_SPLITS
    }
    return make_dataset(context, examples, pairs)


def build_graded_d_dataset(context: ExperimentContext, encoder: Any) -> Dataset:
    examples = build_graded_d_bank(
        encoder,
        fit_contents=16,
        cal_contents=8,
        test_contents=8,
        content_offset=17000,
        q_grid=(0, 1, 2, 4),
    )
    pair_limits = {FIT_SPLIT: 64, CAL_SPLIT: 48, TEST_SPLIT: 64}
    pairs = {
        split: build_graded_pairs(
            examples,
            split=split,
            records_per_relation=pair_limits[split],
            # Compatibility name from the graded_evidence module.
            e_definition=D_DEFINITION,
        )
        for split in ALL_SPLITS
    }
    return make_dataset(context, examples, pairs)


def unlock_and_load_heldout(
    context: ExperimentContext,
    datasets: Sequence[Dataset],
) -> None:
    """This is the only function allowed to materialize Dte activations."""

    if context.heldout_unlocked:
        raise RuntimeError("Dte has already been unlocked")

    context.heldout_unlocked = True
    for dataset in datasets:
        test_examples = tuple(
            example
            for example in dataset.examples
            if example.split == TEST_SPLIT
        )
        dataset.runs.update(collect_runs(context, test_examples))


def ensure_split_is_available(context: ExperimentContext, split: str) -> None:
    if split == TEST_SPLIT and not context.heldout_unlocked:
        raise RuntimeError("Dte is locked until final heldout certification")


def evaluate_outputs(
    context: ExperimentContext,
    dataset: Dataset,
    configurations: Sequence[HandleConfiguration],
    split: str,
) -> np.ndarray:
    ensure_split_is_available(context, split)
    return evaluate_configurations(
        context.model,
        configurations,
        dataset.pairs[split],
        examples=dataset.examples_by_id,
        runs=dataset.runs,
        site_lookup=context.sites_by_id,
        disabled_sites=(),
        hook_means={},
        negative_token_id=context.negative_token_id,
        positive_token_id=context.positive_token_id,
        device=context.device,
        max_batch_size=context.args.max_batch_size,
    )


def evaluate_probes(
    context: ExperimentContext,
    dataset: Dataset,
    configurations: Sequence[HandleConfiguration],
    split: str,
    probe_sites: Sequence[ChannelSite],
    restore_site_ids: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray]:
    ensure_split_is_available(context, split)
    return evaluate_progressive_configurations(
        context.model,
        configurations,
        dataset.pairs[split],
        examples=dataset.examples_by_id,
        runs=dataset.runs,
        site_lookup=context.sites_by_id,
        probe_sites=probe_sites,
        negative_token_id=context.negative_token_id,
        positive_token_id=context.positive_token_id,
        device=context.device,
        max_batch_size=context.args.max_batch_size,
        restore_probe_site_ids=restore_site_ids,
    )


# ---------------------------------------------------------------------------
# Generic handle helpers
# ---------------------------------------------------------------------------


def handle_value(run: ClampedRun, weights: Weights) -> float:
    return sum(
        float(weight) * float(run.features_by_site[site_id])
        for site_id, weight in weights.items()
    )


def handle_order(weights: Weights) -> tuple[int, int]:
    """Return the position of the latest site in a singleton/pair handle."""

    return max(layer_order(site_id) for site_id in weights)


def earlier_sites(
    sites: Sequence[ChannelSite],
    downstream_weights: Weights,
) -> tuple[ChannelSite, ...]:
    downstream_position = handle_order(downstream_weights)
    return tuple(
        site
        for site in sites
        if site.site_id not in downstream_weights
        and layer_order(site.site_id) < downstream_position
    )


def probe_sites_for(
    context: ExperimentContext,
    *handles: Weights,
) -> tuple[ChannelSite, ...]:
    site_ids: list[str] = []
    for handle in handles:
        for site_id in handle:
            if site_id not in site_ids:
                site_ids.append(site_id)
    return tuple(context.sites_by_id[site_id] for site_id in site_ids)


def combine_probe_values(
    probe_values: np.ndarray,
    probe_site_ids: Sequence[str],
    weights: Weights,
) -> np.ndarray:
    probe_index = {
        site_id: position for position, site_id in enumerate(probe_site_ids)
    }
    return sum(
        float(weight) * probe_values[..., probe_index[site_id]]
        for site_id, weight in weights.items()
    )


def normalize_signature_blocks(
    abstract_signature_values: Sequence[float],
    neural_signatures: Mapping[str, Sequence[float]],
    pairs: Sequence[RediscoveryPair],
    components_per_pair: int,
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], dict[str, float]]:
    abstract = torch.tensor(abstract_signature_values, dtype=torch.float32)
    site_ids = tuple(neural_signatures)
    neural = torch.tensor(
        [neural_signatures[site_id] for site_id in site_ids],
        dtype=torch.float32,
    )

    scales: dict[str, float] = {}
    relations = sorted({pair.relation for pair in pairs})
    for relation in relations:
        for component in range(components_per_pair):
            indices = [
                components_per_pair * pair_index + component
                for pair_index, pair in enumerate(pairs)
                if pair.relation == relation
            ]
            abstract_rms = float(torch.sqrt((abstract[indices] ** 2).mean()))
            neural_rms = float(torch.sqrt((neural[:, indices] ** 2).mean()))
            scale = max(abstract_rms, neural_rms, 1e-6)

            abstract[indices] /= scale
            neural[:, indices] /= scale
            scales[f"{relation}:{component}"] = scale

    normalized_neural = {
        site_id: tuple(float(value) for value in neural[index])
        for index, site_id in enumerate(site_ids)
    }
    normalized_abstract = tuple(float(value) for value in abstract)
    return normalized_abstract, normalized_neural, scales


def select_effective_support(
    ranked_sites: Sequence[Mapping[str, Any]],
    top_n: int,
    mass_fraction: float,
) -> list[JsonDict]:
    support = [dict(row) for row in ranked_sites[:top_n]]
    if not support:
        return []

    minimum_weight = mass_fraction * float(support[0]["weight"])
    return [
        row for row in support if float(row["weight"]) >= minimum_weight
    ]


def make_calibration_grid(
    support: Sequence[Mapping[str, Any]],
    strengths: Sequence[float],
) -> tuple[tuple[HandleConfiguration, ...], list[JsonDict]]:
    """Create every singleton/pair handle at every requested strength."""

    configurations: list[HandleConfiguration] = []
    metadata: list[JsonDict] = []

    for handle_size in (1, 2):
        for handle_index, sites in enumerate(
            combinations(support, handle_size),
            start=1,
        ):
            total_mass = sum(float(site["weight"]) for site in sites)
            weights = {
                str(site["site_id"]): float(site["weight"]) / total_mass
                for site in sites
            }

            for strength in strengths:
                handle_id = (
                    f"S{handle_size}_{handle_index}_lambda{strength:g}"
                )
                configurations.append(
                    HandleConfiguration(handle_id, weights, strength)
                )
                metadata.append(
                    {
                        "handle_id": handle_id,
                        "k": handle_size,
                        "strength": strength,
                        "weights": weights,
                        "site_ids": list(weights),
                    }
                )

    return tuple(configurations), metadata


def best_calibration_tier(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    best_score = max(float(row["summary"]["score"]) for row in rows)
    tied_rows = [
        dict(row)
        for row in rows
        if abs(float(row["summary"]["score"]) - best_score) < 1e-12
    ]
    selected_row = dict(select_calibration_row(tied_rows))
    return selected_row, tied_rows


def best_strength_for_each_handle(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep one Dcal strength for each unique singleton/pair support."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        groups[tuple(row["site_ids"])].append(row)

    selected: list[JsonDict] = []
    for same_support_rows in groups.values():
        best = min(
            same_support_rows,
            key=lambda row: (
                -float(row["summary"]["score"]),
                abs(float(row["strength"]) - 1.0),
            ),
        )
        if best["summary"]["passes"]:
            selected.append(dict(best))
    return selected


# ---------------------------------------------------------------------------
# Decode and classify a handle as binary R or graded D
# ---------------------------------------------------------------------------


def fit_r_readout(
    dataset: Dataset,
    weights: Weights,
    report_splits: Sequence[str] = SELECTION_SPLITS,
) -> tuple[Any, dict[str, float]]:
    fit_examples = [
        example for example in dataset.examples if example.split == FIT_SPLIT
    ]
    fit_values = [
        handle_value(dataset.runs[example.example_id], weights)
        for example in fit_examples
    ]
    fit_targets = [example.variable_value for example in fit_examples]
    readout = fit_binary_scalar_readout(fit_values, fit_targets)

    accuracy: dict[str, float] = {}
    for split in report_splits:
        split_examples = [
            example for example in dataset.examples if example.split == split
        ]
        correct = [
            readout.predict(
                handle_value(dataset.runs[example.example_id], weights)
            )
            == example.variable_value
            for example in split_examples
        ]
        accuracy[split] = float(np.mean(correct))
    return readout, accuracy


def fit_d_decoder(
    dataset: Dataset,
    weights: Weights,
    report_splits: Sequence[str] = SELECTION_SPLITS,
) -> tuple[Any, dict[str, dict[str, float]]]:
    fit_examples = [
        example for example in dataset.examples if example.split == FIT_SPLIT
    ]
    fit_values = [
        handle_value(dataset.runs[example.example_id], weights)
        for example in fit_examples
    ]
    fit_targets = [d_value(example, D_DEFINITION) for example in fit_examples]
    decoder = fit_affine_decoder(fit_values, fit_targets)

    metrics: dict[str, dict[str, float]] = {}
    for split in report_splits:
        split_examples = [
            example for example in dataset.examples if example.split == split
        ]
        values = [
            handle_value(dataset.runs[example.example_id], weights)
            for example in split_examples
        ]
        targets = [
            d_value(example, D_DEFINITION) for example in split_examples
        ]
        metrics[split] = decoder_metrics(decoder, values, targets)
    return decoder, metrics


def metrics_look_like_d(
    metrics: Mapping[str, Mapping[str, float]],
    pearson_threshold: float,
) -> bool:
    selection_pearsons = [
        abs(float(metrics[split]["pearson"])) for split in SELECTION_SPLITS
    ]
    return min(selection_pearsons) >= pearson_threshold


def classify_calibrated_handles(
    calibration_rows: Sequence[Mapping[str, Any]],
    graded_dataset: Dataset,
    d_pearson_threshold: float,
) -> list[HandleCandidate]:
    """Fit all readouts on Dfit and report them on Dfit/Dcal only."""

    candidates: list[HandleCandidate] = []
    seen_weights: set[tuple[tuple[str, float], ...]] = set()

    for calibration in calibration_rows:
        weights = {
            str(site_id): float(weight)
            for site_id, weight in calibration["weights"].items()
        }
        weight_key = tuple(sorted(weights.items()))
        if weight_key in seen_weights:
            continue
        seen_weights.add(weight_key)

        r_readout, r_accuracy = fit_r_readout(graded_dataset, weights)
        d_decoder, d_metrics = fit_d_decoder(graded_dataset, weights)
        candidates.append(
            HandleCandidate(
                calibration=dict(calibration),
                weights=weights,
                r_readout=r_readout,
                r_accuracy=r_accuracy,
                d_decoder=d_decoder,
                d_metrics=d_metrics,
                is_d=metrics_look_like_d(
                    d_metrics,
                    d_pearson_threshold,
                ),
            )
        )
    return candidates


def public_handle(candidate: HandleCandidate) -> JsonDict:
    return {
        "handle_id": candidate.calibration["handle_id"],
        "weights": candidate.weights,
        "strength": candidate.calibration["strength"],
        "R_accuracy": candidate.r_accuracy,
        "D_decoder": candidate.d_decoder.to_dict(),
        "D_metrics": candidate.d_metrics,
        "is_D": candidate.is_d,
    }


# ---------------------------------------------------------------------------
# Causal summaries
# ---------------------------------------------------------------------------


def expected_r_after_patch(
    pair: RediscoveryPair,
    examples_by_id: Mapping[str, RediscoveryExample],
) -> int:
    base_r = examples_by_id[pair.base_id].variable_value
    source_r = examples_by_id[pair.source_id].variable_value
    return int(source_r if source_r != base_r else base_r)


def summarize_r_intervention(
    dataset: Dataset,
    split: str,
    output_margins: Sequence[float],
    downstream_r_values: Sequence[float],
    frozen_r: HandleCandidate,
) -> JsonDict:
    """Measure sensitivity and invariance for an upstream -> frozen R edge."""

    rows_by_relation: dict[str, list[dict[str, bool]]] = defaultdict(list)
    sensitivity_rows: list[dict[str, bool]] = []
    invariance_rows: list[dict[str, bool]] = []

    for pair_index, pair in enumerate(dataset.pairs[split]):
        base_value = handle_value(dataset.runs[pair.base_id], frozen_r.weights)
        source_value = handle_value(
            dataset.runs[pair.source_id],
            frozen_r.weights,
        )
        patched_value = float(downstream_r_values[pair_index])
        expected_r = expected_r_after_patch(pair, dataset.examples_by_id)

        source_differs = (
            dataset.examples_by_id[pair.source_id].variable_value
            != dataset.examples_by_id[pair.base_id].variable_value
        )
        if abs(source_value - base_value) > 1e-8:
            downstream_moves = abs(source_value - patched_value) < abs(
                source_value - base_value
            )
        else:
            downstream_moves = abs(patched_value - base_value) <= 1e-6

        row = {
            "output_correct": (
                1 if float(output_margins[pair_index]) > 0 else -1
            )
            == expected_r,
            "downstream_correct": frozen_r.r_readout.predict(patched_value)
            == expected_r,
            "downstream_moves": downstream_moves,
        }
        rows_by_relation[pair.relation].append(row)
        if source_differs:
            sensitivity_rows.append(row)
        else:
            invariance_rows.append(row)

    metric_names = (
        "output_correct",
        "downstream_correct",
        "downstream_moves",
    )
    relations = {
        relation: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in metric_names
        }
        for relation, rows in sorted(rows_by_relation.items())
    }
    balanced_blocks = {
        "sensitivity_output": float(
            np.mean([row["output_correct"] for row in sensitivity_rows])
        ),
        "sensitivity_downstream": float(
            np.mean([row["downstream_correct"] for row in sensitivity_rows])
        ),
        "invariance_output": float(
            np.mean([row["output_correct"] for row in invariance_rows])
        ),
        "invariance_downstream": float(
            np.mean([row["downstream_correct"] for row in invariance_rows])
        ),
    }
    return {
        "relations": relations,
        "balanced_blocks": balanced_blocks,
        "score": float(np.mean(list(balanced_blocks.values()))),
        "passes": min(balanced_blocks.values()) >= PASS_THRESHOLD,
    }


def summarize_d_intervention(
    dataset: Dataset,
    split: str,
    output_margins: Sequence[float],
    probe_values: np.ndarray,
    probe_site_ids: Sequence[str],
    frozen_d: HandleCandidate,
    frozen_r: HandleCandidate,
) -> JsonDict:
    abstract_d = {
        example_id: d_value(dataset.examples_by_id[example_id], D_DEFINITION)
        for example_id in dataset.runs
    }
    clean_d = {
        example_id: frozen_d.d_decoder.predict(
            handle_value(run, frozen_d.weights)
        )
        for example_id, run in dataset.runs.items()
    }
    patched_d_values = combine_probe_values(
        probe_values,
        probe_site_ids,
        frozen_d.weights,
    )
    patched_d = (
        frozen_d.d_decoder.slope * patched_d_values
        + frozen_d.d_decoder.intercept
    )
    patched_r_values = combine_probe_values(
        probe_values,
        probe_site_ids,
        frozen_r.weights,
    )
    patched_r = [
        frozen_r.r_readout.predict(value) for value in patched_r_values
    ]
    patched_y = np.where(np.asarray(output_margins) > 0, 1, -1)

    return graded_validation_summary(
        dataset.pairs[split],
        dataset.examples_by_id,
        abstract_d,
        clean_d,
        patched_d,
        patched_r,
        patched_y,
    )


def certification_passes(summary: JsonDict, mediation: JsonDict) -> bool:
    return bool(summary["passes"] and mediation["passes"])


# ---------------------------------------------------------------------------
# Dfit ranking and Dcal calibration
# ---------------------------------------------------------------------------


def coarse_r_search(
    context: ExperimentContext,
    coarse_dataset: Dataset,
    strengths: Sequence[float],
) -> JsonDict:
    """Rank all sites on Dfit, then calibrate support handles on Dcal."""

    print("[1] Coarse R search over all candidate sites", flush=True)

    singleton_configs = tuple(
        HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0)
        for site in context.sites
    )
    fit_margins = evaluate_outputs(
        context,
        coarse_dataset,
        singleton_configs,
        FIT_SPLIT,
    )

    neural_signatures: dict[str, tuple[float, ...]] = {}
    for config_index, config in enumerate(singleton_configs):
        effects = []
        for pair_index, pair in enumerate(coarse_dataset.pairs[FIT_SPLIT]):
            clean_margin = coarse_dataset.runs[pair.base_id].class_margin
            effects.append(
                float(fit_margins[config_index, pair_index] - clean_margin)
            )
        neural_signatures[config.handle_id] = tuple(effects)

    abstract = abstract_signature(
        coarse_dataset.pairs[FIT_SPLIT],
        coarse_dataset.examples_by_id,
    )
    abstract, neural_signatures, scales = normalize_signature_blocks(
        abstract,
        neural_signatures,
        coarse_dataset.pairs[FIT_SPLIT],
        components_per_pair=1,
    )
    selector = match_signatures(
        abstract,
        neural_signatures,
        epsilon=context.args.selector_epsilon,
        beta=context.args.selector_beta,
    )
    support = select_effective_support(
        selector["ranked"],
        context.args.top_n,
        context.args.mass_fraction,
    )

    configurations, metadata = make_calibration_grid(support, strengths)
    cal_margins = evaluate_outputs(
        context,
        coarse_dataset,
        configurations,
        CAL_SPLIT,
    )
    calibration_rows = [
        {
            **metadata[index],
            "summary": relation_summary(
                coarse_dataset.pairs[CAL_SPLIT],
                coarse_dataset.examples_by_id,
                cal_margins[index],
            ),
        }
        for index in range(len(metadata))
    ]
    best, best_tier = best_calibration_tier(calibration_rows)

    return {
        "normalization_scales": scales,
        "selector": selector,
        "effective_support": support,
        "calibration_best": best,
        "calibration_best_tier": best_tier,
        "heldout": [],
        "accepted": [],
    }


def upstream_signature(
    dataset: Dataset,
    split: str,
    downstream_values: Sequence[float],
    frozen_weights: Weights,
    scale: float,
) -> tuple[float, ...]:
    signature = []
    for pair_index, pair in enumerate(dataset.pairs[split]):
        base_value = handle_value(dataset.runs[pair.base_id], frozen_weights)
        patched_value = float(downstream_values[pair_index])
        signature.append(scale * (patched_value - base_value))
    return tuple(signature)


def target_r_signature(
    dataset: Dataset,
    split: str,
) -> tuple[float, ...]:
    return tuple(
        float(
            dataset.examples_by_id[pair.source_id].variable_value
            - dataset.examples_by_id[pair.base_id].variable_value
        )
        for pair in dataset.pairs[split]
    )


def rank_sites_against_frozen_r(
    context: ExperimentContext,
    dataset: Dataset,
    candidate_sites: Sequence[ChannelSite],
    frozen_r: HandleCandidate,
) -> JsonDict:
    probe_sites = probe_sites_for(context, frozen_r.weights)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    configurations = tuple(
        HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0)
        for site in candidate_sites
    )
    output_margins, probe_values = evaluate_probes(
        context,
        dataset,
        configurations,
        FIT_SPLIT,
        probe_sites,
    )

    del output_margins  # Ranking this edge uses movement at the frozen R only.
    neural_signatures = {}
    for index, configuration in enumerate(configurations):
        downstream_r = combine_probe_values(
            probe_values[index],
            probe_site_ids,
            frozen_r.weights,
        )
        neural_signatures[configuration.handle_id] = upstream_signature(
            dataset,
            FIT_SPLIT,
            downstream_r,
            frozen_r.weights,
            scale=frozen_r.r_readout.orientation,
        )

    selector = match_signatures(
        target_r_signature(dataset, FIT_SPLIT),
        neural_signatures,
        epsilon=context.args.selector_epsilon,
        beta=context.args.selector_beta,
    )
    return {"selector": selector, "normalization_scales": {}}


def calibrate_against_frozen_r(
    context: ExperimentContext,
    dataset: Dataset,
    ranked_sites: Sequence[Mapping[str, Any]],
    frozen_r: HandleCandidate,
    strengths: Sequence[float],
) -> JsonDict:
    probe_sites = probe_sites_for(context, frozen_r.weights)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    configurations, metadata = make_calibration_grid(ranked_sites, strengths)

    output_margins, probe_values = evaluate_probes(
        context,
        dataset,
        configurations,
        CAL_SPLIT,
        probe_sites,
    )
    downstream_r = combine_probe_values(
        probe_values,
        probe_site_ids,
        frozen_r.weights,
    )
    calibration_rows = [
        {
            **metadata[index],
            "summary": summarize_r_intervention(
                dataset,
                CAL_SPLIT,
                output_margins[index],
                downstream_r[index],
                frozen_r,
            ),
        }
        for index in range(len(metadata))
    ]
    best, best_tier = best_calibration_tier(calibration_rows)

    return {
        "calibration_best": best,
        "calibration_best_tier": best_tier,
        "calibrated_handles": best_strength_for_each_handle(calibration_rows),
        "heldout": [],
        "accepted": [],
    }


def rank_sites_against_frozen_d(
    context: ExperimentContext,
    dataset: Dataset,
    candidate_sites: Sequence[ChannelSite],
    frozen_d: HandleCandidate,
) -> JsonDict:
    probe_sites = probe_sites_for(context, frozen_d.weights)
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    configurations = tuple(
        HandleConfiguration(site.site_id, {site.site_id: 1.0}, 1.0)
        for site in candidate_sites
    )
    output_margins, probe_values = evaluate_probes(
        context,
        dataset,
        configurations,
        FIT_SPLIT,
        probe_sites,
    )

    del output_margins  # Ranking this edge uses movement at the frozen D only.
    neural_signatures = {}
    for index, configuration in enumerate(configurations):
        downstream_d = combine_probe_values(
            probe_values[index],
            probe_site_ids,
            frozen_d.weights,
        )
        neural_signatures[configuration.handle_id] = upstream_signature(
            dataset,
            FIT_SPLIT,
            downstream_d,
            frozen_d.weights,
            scale=frozen_d.d_decoder.slope,
        )

    abstract = abstract_d_signature(
        dataset.pairs[FIT_SPLIT],
        dataset.examples_by_id,
        definition=D_DEFINITION,
    )
    abstract, neural_signatures, scales = normalize_signature_blocks(
        abstract,
        neural_signatures,
        dataset.pairs[FIT_SPLIT],
        components_per_pair=1,
    )
    selector = match_signatures(
        abstract,
        neural_signatures,
        epsilon=context.args.selector_epsilon,
        beta=context.args.selector_beta,
    )
    return {"selector": selector, "normalization_scales": scales}


def calibrate_against_frozen_d(
    context: ExperimentContext,
    dataset: Dataset,
    ranked_sites: Sequence[Mapping[str, Any]],
    frozen_d: HandleCandidate,
    frozen_r: HandleCandidate,
    strengths: Sequence[float],
) -> JsonDict:
    probe_sites = probe_sites_for(
        context,
        frozen_d.weights,
        frozen_r.weights,
    )
    probe_site_ids = tuple(site.site_id for site in probe_sites)
    configurations, metadata = make_calibration_grid(ranked_sites, strengths)
    output_margins, probe_values = evaluate_probes(
        context,
        dataset,
        configurations,
        CAL_SPLIT,
        probe_sites,
    )

    calibration_rows = [
        {
            **metadata[index],
            "summary": summarize_d_intervention(
                dataset,
                CAL_SPLIT,
                output_margins[index],
                probe_values[index],
                probe_site_ids,
                frozen_d,
                frozen_r,
            ),
        }
        for index in range(len(metadata))
    ]
    best, best_tier = best_calibration_tier(calibration_rows)

    return {
        "calibration_best": best,
        "calibration_best_tier": best_tier,
        "calibrated_handles": best_strength_for_each_handle(calibration_rows),
        "heldout": [],
        "accepted": [],
    }


# ---------------------------------------------------------------------------
# Explicit selection rules
# ---------------------------------------------------------------------------


def is_valid_r(candidate: HandleCandidate) -> bool:
    return (
        not candidate.is_d
        and min(candidate.r_accuracy.values()) >= PASS_THRESHOLD
    )


def choose_late_r(
    candidates: Sequence[HandleCandidate],
    coarse_site_rank: Mapping[str, int],
) -> HandleCandidate | None:
    """Prefer smaller K, then sites with better coarse OT rank."""

    valid = [candidate for candidate in candidates if is_valid_r(candidate)]
    if not valid:
        return None

    def priority(candidate: HandleCandidate) -> tuple[int, int]:
        best_site_rank = min(
            coarse_site_rank.get(site_id, 10**9)
            for site_id in candidate.weights
        )
        return int(candidate.calibration["k"]), best_site_rank

    return min(valid, key=priority)


def choose_refined_r(
    candidates: Sequence[HandleCandidate],
) -> HandleCandidate | None:
    """Among valid R handles, prefer earlier position, then smaller K."""

    valid = [candidate for candidate in candidates if is_valid_r(candidate)]
    if not valid:
        return None

    def priority(candidate: HandleCandidate) -> tuple[tuple[int, int], int]:
        return handle_order(candidate.weights), int(candidate.calibration["k"])

    return min(valid, key=priority)


def choose_late_d(
    candidates: Sequence[HandleCandidate],
) -> HandleCandidate | None:
    """Choose the strongest Dcal D handle; use simple tie breakers."""

    valid = [candidate for candidate in candidates if candidate.is_d]
    if not valid:
        return None

    def priority(candidate: HandleCandidate) -> tuple[float, int, float, float]:
        calibration = candidate.calibration
        return (
            float(calibration["summary"]["score"]),
            -int(calibration["k"]),
            -abs(float(calibration["strength"]) - 1.0),
            abs(float(candidate.d_metrics[CAL_SPLIT]["pearson"])),
        )

    return max(valid, key=priority)


def choose_refined_d(
    candidates: Sequence[HandleCandidate],
) -> HandleCandidate | None:
    """Dcal causal score dominates; depth/order is only a later tie breaker."""

    valid = [candidate for candidate in candidates if candidate.is_d]
    if not valid:
        return None

    def priority(candidate: HandleCandidate) -> tuple[Any, ...]:
        calibration = candidate.calibration
        dcal_metrics = candidate.d_metrics[CAL_SPLIT]
        return (
            -float(calibration["summary"]["score"]),
            -abs(float(dcal_metrics["pearson"])),
            float(dcal_metrics["mae"]),
            handle_order(candidate.weights),
            int(calibration["k"]),
            abs(float(calibration["strength"]) - 1.0),
        )

    return min(valid, key=priority)


def label_chain(chain: list[JsonDict], variable: str) -> None:
    if len(chain) == 1:
        chain[0]["name"] = variable
        return

    chain[0]["name"] = f"{variable}_late"
    chain[-1]["name"] = f"{variable}_early"
    for index in range(1, len(chain) - 1):
        chain[index]["name"] = f"{variable}_mid_{len(chain) - index - 1}"


# ---------------------------------------------------------------------------
# Discovery phases: Dfit/Dcal only
# ---------------------------------------------------------------------------


def discover_r_chain(
    context: ExperimentContext,
    coarse_dataset: Dataset,
    graded_dataset: Dataset,
    strengths: Sequence[float],
) -> RDiscovery:
    coarse_result = coarse_r_search(context, coarse_dataset, strengths)

    coarse_candidates = classify_calibrated_handles(
        coarse_result["calibration_best_tier"],
        graded_dataset,
        context.args.graded_threshold,
    )
    coarse_result["classified_handles"] = [
        public_handle(candidate) for candidate in coarse_candidates
    ]
    site_rank = {
        str(row["site_id"]): index
        for index, row in enumerate(coarse_result["selector"]["ranked"])
    }
    late_r = choose_late_r(coarse_candidates, site_rank)
    if late_r is None:
        raise RuntimeError(
            "No binary R handle passed coarse Dfit/Dcal selection"
        )

    support_sites = tuple(
        context.sites_by_id[str(row["site_id"])]
        for row in coarse_result["effective_support"]
    )
    refinement_sites = earlier_sites(support_sites, late_r.weights)

    early_r: HandleCandidate | None = None
    refinement: JsonDict | None = None
    if refinement_sites:
        print(
            f"[2] R refinement: {len(refinement_sites)} upstream candidates",
            flush=True,
        )
        ranking = rank_sites_against_frozen_r(
            context,
            coarse_dataset,
            refinement_sites,
            late_r,
        )
        calibration = calibrate_against_frozen_r(
            context,
            coarse_dataset,
            ranking["selector"]["ranked"],
            late_r,
            strengths,
        )
        candidates = classify_calibrated_handles(
            calibration["calibrated_handles"],
            graded_dataset,
            context.args.graded_threshold,
        )
        early_r = choose_refined_r(candidates)
        refinement = {
            "candidate_ids": [site.site_id for site in refinement_sites],
            **ranking,
            **calibration,
            "classified_handles": [
                public_handle(candidate) for candidate in candidates
            ],
        }

    chain = [
        {
            "name": "R",
            **public_handle(late_r),
            "source": "coarse Dfit/Dcal search",
        }
    ]
    if early_r is not None:
        chain.append(
            {
                "name": "R_early",
                **public_handle(early_r),
                "source": "R refinement on Dfit/Dcal",
            }
        )
    label_chain(chain, "R")

    return RDiscovery(
        coarse_result=coarse_result,
        late=late_r,
        early=early_r,
        refinement=refinement,
        chain=chain,
    )


def evaluate_d_edge(
    context: ExperimentContext,
    dataset: Dataset,
    source_calibration: Mapping[str, Any],
    frozen_d: HandleCandidate,
    frozen_r: HandleCandidate,
    split: str,
    restore_weights: Weights,
) -> JsonDict:
    """Intervene on source, observe frozen D/R/Y, then run restoration."""

    configuration = HandleConfiguration(
        source_calibration["handle_id"],
        source_calibration["weights"],
        source_calibration["strength"],
    )
    probe_sites = probe_sites_for(
        context,
        frozen_d.weights,
        frozen_r.weights,
    )
    probe_site_ids = tuple(site.site_id for site in probe_sites)

    output_margins, probe_values = evaluate_probes(
        context,
        dataset,
        (configuration,),
        split,
        probe_sites,
    )
    restored_margins, _ = evaluate_probes(
        context,
        dataset,
        (configuration,),
        split,
        probe_sites,
        restore_site_ids=tuple(restore_weights),
    )
    summary = summarize_d_intervention(
        dataset,
        split,
        output_margins[0],
        probe_values[0],
        probe_site_ids,
        frozen_d,
        frozen_r,
    )
    mediation = mediation_summary(
        dataset.pairs[split],
        dataset.examples_by_id,
        dataset.runs,
        output_margins[0],
        restored_margins[0],
    )
    return {
        "summary": summary,
        "R_mediation": mediation,
        "accepted": certification_passes(summary, mediation),
    }


def validate_late_d_on_dcal(
    context: ExperimentContext,
    graded_dataset: Dataset,
    late_d: HandleCandidate,
    final_r: HandleCandidate,
) -> JsonDict:
    dcal_result = evaluate_d_edge(
        context,
        graded_dataset,
        late_d.calibration,
        frozen_d=late_d,
        frozen_r=final_r,
        split=CAL_SPLIT,
        restore_weights=final_r.weights,
    )
    if not dcal_result["accepted"]:
        raise RuntimeError("Selected D handle failed Dcal D -> R -> Y validation")

    return {
        "weights": late_d.weights,
        "decoder": late_d.d_decoder.to_dict(),
        "metrics": late_d.d_metrics,
        "R_readout": final_r.r_readout.to_dict(),
        "R_accuracy": final_r.r_accuracy,
        "splits": {CAL_SPLIT: dcal_result},
    }


def discover_d_chain(
    context: ExperimentContext,
    graded_dataset: Dataset,
    final_r: HandleCandidate,
    strengths: Sequence[float],
) -> DDiscovery:
    candidate_sites = earlier_sites(context.sites, final_r.weights)
    if not candidate_sites:
        raise RuntimeError("No site is earlier than the final R handle")

    print(
        f"[3] D discovery: {len(candidate_sites)} upstream candidates",
        flush=True,
    )
    ranking = rank_sites_against_frozen_r(
        context,
        graded_dataset,
        candidate_sites,
        final_r,
    )
    support = select_effective_support(
        ranking["selector"]["ranked"],
        context.args.top_n,
        context.args.mass_fraction,
    )
    calibration = calibrate_against_frozen_r(
        context,
        graded_dataset,
        support,
        final_r,
        strengths,
    )
    candidates = classify_calibrated_handles(
        calibration["calibrated_handles"],
        graded_dataset,
        context.args.graded_threshold,
    )
    late_d = choose_late_d(candidates)
    if late_d is None:
        raise RuntimeError("No graded D handle passed Dfit/Dcal discovery")

    discovery_result = {
        "candidate_ids": [site.site_id for site in candidate_sites],
        **ranking,
        "effective_support": support,
        **calibration,
        "classified_handles": [
            public_handle(candidate) for candidate in candidates
        ],
    }
    direct_validation = validate_late_d_on_dcal(
        context,
        graded_dataset,
        late_d,
        final_r,
    )

    support_sites = tuple(
        context.sites_by_id[str(row["site_id"])] for row in support
    )
    refinement_sites = tuple(
        site
        for site in earlier_sites(support_sites, late_d.weights)
        if site.site_id not in final_r.weights
    )

    early_d: HandleCandidate | None = None
    refinement: JsonDict | None = None
    if refinement_sites:
        print(
            f"[4] D refinement: {len(refinement_sites)} upstream candidates",
            flush=True,
        )
        refinement_ranking = rank_sites_against_frozen_d(
            context,
            graded_dataset,
            refinement_sites,
            late_d,
        )
        refinement_calibration = calibrate_against_frozen_d(
            context,
            graded_dataset,
            refinement_ranking["selector"]["ranked"],
            late_d,
            final_r,
            strengths,
        )
        refinement_candidates = classify_calibrated_handles(
            refinement_calibration["calibrated_handles"],
            graded_dataset,
            context.args.graded_threshold,
        )
        early_d = choose_refined_d(refinement_candidates)
        refinement = {
            "candidate_ids": [site.site_id for site in refinement_sites],
            **refinement_ranking,
            **refinement_calibration,
            "classified_handles": [
                public_handle(candidate) for candidate in refinement_candidates
            ],
        }

    chain = [
        {
            "name": "D",
            **public_handle(late_d),
            "source": "independent D discovery on Dfit/Dcal",
        }
    ]
    if early_d is not None:
        chain.append(
            {
                "name": "D_early",
                **public_handle(early_d),
                "source": "D refinement on Dfit/Dcal",
            }
        )
    label_chain(chain, "D")

    return DDiscovery(
        discovery_result=discovery_result,
        direct_validation=direct_validation,
        late=late_d,
        early=early_d,
        refinement=refinement,
        chain=chain,
    )


# ---------------------------------------------------------------------------
# Final heldout certification: Dte only
# ---------------------------------------------------------------------------


def certify_r_to_output_on_dte(
    context: ExperimentContext,
    coarse_dataset: Dataset,
    late_r: HandleCandidate,
) -> JsonDict:
    calibration = late_r.calibration
    configuration = HandleConfiguration(
        calibration["handle_id"],
        calibration["weights"],
        calibration["strength"],
    )
    output_margins = evaluate_outputs(
        context,
        coarse_dataset,
        (configuration,),
        TEST_SPLIT,
    )[0]
    summary = relation_summary(
        coarse_dataset.pairs[TEST_SPLIT],
        coarse_dataset.examples_by_id,
        output_margins,
    )
    return {
        **calibration,
        "heldout": summary,
        "accepted": bool(summary["all_rates_at_least_0_90"]),
    }


def certify_r_refinement_on_dte(
    context: ExperimentContext,
    coarse_dataset: Dataset,
    early_r: HandleCandidate,
    late_r: HandleCandidate,
) -> JsonDict:
    calibration = early_r.calibration
    configuration = HandleConfiguration(
        calibration["handle_id"],
        calibration["weights"],
        calibration["strength"],
    )
    probe_sites = probe_sites_for(context, late_r.weights)
    probe_site_ids = tuple(site.site_id for site in probe_sites)

    output_margins, probe_values = evaluate_probes(
        context,
        coarse_dataset,
        (configuration,),
        TEST_SPLIT,
        probe_sites,
    )
    restored_margins, _ = evaluate_probes(
        context,
        coarse_dataset,
        (configuration,),
        TEST_SPLIT,
        probe_sites,
        restore_site_ids=probe_site_ids,
    )
    downstream_r = combine_probe_values(
        probe_values[0],
        probe_site_ids,
        late_r.weights,
    )
    summary = summarize_r_intervention(
        coarse_dataset,
        TEST_SPLIT,
        output_margins[0],
        downstream_r,
        late_r,
    )
    mediation = mediation_summary(
        coarse_dataset.pairs[TEST_SPLIT],
        coarse_dataset.examples_by_id,
        coarse_dataset.runs,
        output_margins[0],
        restored_margins[0],
    )
    return {
        **calibration,
        "heldout": summary,
        "mediation": mediation,
        "accepted": certification_passes(summary, mediation),
    }


def certify_final_model(
    context: ExperimentContext,
    coarse_dataset: Dataset,
    graded_dataset: Dataset,
    r_discovery: RDiscovery,
    d_discovery: DDiscovery,
) -> tuple[JsonDict, bool]:
    """Unlock Dte once, then test only the already selected edges."""

    print("[5] Final heldout certification on Dte", flush=True)
    unlock_and_load_heldout(context, (coarse_dataset, graded_dataset))

    r_to_y = certify_r_to_output_on_dte(
        context,
        coarse_dataset,
        r_discovery.late,
    )
    r_discovery.coarse_result["heldout"] = [r_to_y]
    r_discovery.coarse_result["accepted"] = (
        [r_to_y] if r_to_y["accepted"] else []
    )

    r_early_to_late = None
    if r_discovery.early is not None:
        r_early_to_late = certify_r_refinement_on_dte(
            context,
            coarse_dataset,
            r_discovery.early,
            r_discovery.late,
        )
    if r_discovery.refinement is not None:
        r_discovery.refinement["heldout"] = (
            [r_early_to_late] if r_early_to_late is not None else []
        )
        r_discovery.refinement["accepted"] = (
            [r_early_to_late]
            if r_early_to_late is not None
            and r_early_to_late["accepted"]
            else []
        )

    d_late_to_r = evaluate_d_edge(
        context,
        graded_dataset,
        d_discovery.late.calibration,
        frozen_d=d_discovery.late,
        frozen_r=r_discovery.final,
        split=TEST_SPLIT,
        restore_weights=r_discovery.final.weights,
    )
    d_discovery.direct_validation["splits"][TEST_SPLIT] = d_late_to_r
    _, all_d_metrics = fit_d_decoder(
        graded_dataset,
        d_discovery.late.weights,
        ALL_SPLITS,
    )
    _, all_r_accuracy = fit_r_readout(
        graded_dataset,
        r_discovery.final.weights,
        ALL_SPLITS,
    )
    d_discovery.direct_validation["metrics"] = all_d_metrics
    d_discovery.direct_validation["R_accuracy"] = all_r_accuracy

    d_early_to_late = None
    if d_discovery.early is not None:
        d_early_to_late = evaluate_d_edge(
            context,
            graded_dataset,
            d_discovery.early.calibration,
            frozen_d=d_discovery.late,
            frozen_r=r_discovery.final,
            split=TEST_SPLIT,
            restore_weights=d_discovery.late.weights,
        )
    if d_discovery.refinement is not None:
        d_discovery.refinement["heldout"] = (
            [d_early_to_late] if d_early_to_late is not None else []
        )
        d_discovery.refinement["accepted"] = (
            [d_early_to_late]
            if d_early_to_late is not None
            and d_early_to_late["accepted"]
            else []
        )

    certification = {
        "R_to_Y": r_to_y,
        "R_early_to_R_late": r_early_to_late,
        "D_late_to_R_to_Y": d_late_to_r,
        "D_early_to_D_late": d_early_to_late,
    }
    passed = all(
        edge is None or bool(edge["accepted"])
        for edge in certification.values()
    )
    return certification, passed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def final_model_name(
    d_chain: Sequence[Mapping[str, Any]],
    r_chain: Sequence[Mapping[str, Any]],
) -> str:
    ordered_variables = [
        row["name"] for row in reversed(d_chain)
    ] + [row["name"] for row in reversed(r_chain)]
    return "X -> " + " -> ".join(ordered_variables) + " -> Y"


def build_result(
    context: ExperimentContext,
    model_info: JsonDict,
    sparse_records: Sequence[Any],
    coarse_dataset: Dataset,
    graded_dataset: Dataset,
    r_discovery: RDiscovery,
    d_discovery: DDiscovery,
    certification: JsonDict,
    certification_passed: bool,
) -> JsonDict:
    clean_accuracies = {
        "coarse": {
            split: clean_accuracy(
                coarse_dataset.examples,
                coarse_dataset.runs,
                split=split,
            )
            for split in ALL_SPLITS
        },
        "graded": {
            split: clean_accuracy(
                graded_dataset.examples,
                graded_dataset.runs,
                split=split,
            )
            for split in ALL_SPLITS
        },
    }

    coarse_public = {
        key: value
        for key, value in r_discovery.coarse_result.items()
        if key != "accepted"
    }
    return {
        "experiment": "automatic_gradual_PLOT_bracket",
        "declared_model": (
            "X -> D -> R -> Y, "
            "D=active bracket depth, R=1[D>=2]"
        ),
        "scope": (
            "The code localizes declared variables; "
            "it does not invent new causal variables."
        ),
        "rules": {
            "support": (
                f"top {context.args.top_n}, mass >= "
                f"{context.args.mass_fraction:g} * top-1"
            ),
            "calibration": "all singletons and pairs",
            "D_selection": (
                "Dcal causal score, Dcal Pearson/MAE, then depth/K/strength "
                "tie breakers"
            ),
            "heldout": (
                "Select every handle and hyperparameter on Dfit/Dcal; "
                "unlock Dte only after both chains are frozen"
            ),
        },
        "model_info": model_info,
        "sparse_conversion": [row.to_json() for row in sparse_records],
        "banks": {
            "coarse": bank_manifest(
                coarse_dataset.examples,
                coarse_dataset.pairs,
            ),
            "graded": bank_manifest(
                graded_dataset.examples,
                graded_dataset.pairs,
            ),
        },
        "clean_accuracy": clean_accuracies,
        "coarse": coarse_public,
        "R_refinement": r_discovery.refinement,
        "R_chain": r_discovery.chain,
        "D_discovery": d_discovery.discovery_result,
        "direct_D_to_R_to_Y": d_discovery.direct_validation,
        "D_refinement": d_discovery.refinement,
        "D_chain": d_discovery.chain,
        "final_model": final_model_name(
            d_discovery.chain,
            r_discovery.chain,
        ),
        "final_certification": certification,
        "final_certification_passed": certification_passed,
    }


def format_weights(weights: Weights) -> str:
    """Compact representation used only in the human-readable summary."""

    return ", ".join(
        f"{site_id} ({float(weight):.3f})"
        for site_id, weight in weights.items()
    )


def handle_summary_row(
    name: str,
    candidate: HandleCandidate,
    variable: str,
) -> str:
    calibration = candidate.calibration
    sites = format_weights(candidate.weights)
    strength = float(calibration["strength"])

    if variable == "R":
        evidence = (
            f"R accuracy: Dfit={candidate.r_accuracy[FIT_SPLIT]:.3f}, "
            f"Dcal={candidate.r_accuracy[CAL_SPLIT]:.3f}"
        )
    else:
        dfit = candidate.d_metrics[FIT_SPLIT]
        dcal = candidate.d_metrics[CAL_SPLIT]
        evidence = (
            f"D Pearson: Dfit={float(dfit['pearson']):.3f}, "
            f"Dcal={float(dcal['pearson']):.3f}; "
            f"Dcal MAE={float(dcal['mae']):.3f}"
        )

    return (
        f"| {name} | {calibration['handle_id']} | {sites} | "
        f"{strength:g} | {evidence} |"
    )


def certification_summary_row(edge_name: str, edge: JsonDict | None) -> str:
    if edge is None:
        return f"| {edge_name} | Not present | - | - |"

    status = "PASS" if edge["accepted"] else "FAIL"
    causal_summary = edge.get("heldout", edge.get("summary", {}))
    score = causal_summary.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "-"

    restoration = edge.get("mediation", edge.get("R_mediation"))
    if restoration is None:
        restoration_text = "Not required"
    else:
        restoration_text = "PASS" if restoration["passes"] else "FAIL"

    return f"| {edge_name} | {status} | {score_text} | {restoration_text} |"


def build_readable_summary(
    r_discovery: RDiscovery,
    d_discovery: DDiscovery,
    certification: JsonDict,
    certification_passed: bool,
    detailed_output_path: Path,
) -> str:
    """Create the short Markdown report intended for a human reader."""

    overall_status = "PASS" if certification_passed else "FAIL"
    model = final_model_name(d_discovery.chain, r_discovery.chain)

    handle_rows = [
        handle_summary_row(
            r_discovery.chain[0]["name"],
            r_discovery.late,
            "R",
        )
    ]
    if r_discovery.early is not None:
        handle_rows.append(
            handle_summary_row(
                r_discovery.chain[-1]["name"],
                r_discovery.early,
                "R",
            )
        )
    handle_rows.append(
        handle_summary_row(
            d_discovery.chain[0]["name"],
            d_discovery.late,
            "D",
        )
    )
    if d_discovery.early is not None:
        handle_rows.append(
            handle_summary_row(
                d_discovery.chain[-1]["name"],
                d_discovery.early,
                "D",
            )
        )

    certification_rows = [
        certification_summary_row("R -> Y", certification["R_to_Y"]),
        certification_summary_row(
            "R_early -> R_late",
            certification["R_early_to_R_late"],
        ),
        certification_summary_row(
            "D_late -> R -> Y",
            certification["D_late_to_R_to_Y"],
        ),
        certification_summary_row(
            "D_early -> D_late",
            certification["D_early_to_D_late"],
        ),
    ]

    coarse_count = len(r_discovery.coarse_result["selector"]["ranked"])
    d_count = len(d_discovery.discovery_result["candidate_ids"])
    r_refinement_count = (
        len(r_discovery.refinement["candidate_ids"])
        if r_discovery.refinement is not None
        else 0
    )
    d_refinement_count = (
        len(d_discovery.refinement["candidate_ids"])
        if d_discovery.refinement is not None
        else 0
    )

    lines = [
        "# Automatic Gradual Discovery Summary",
        "",
        "## Final result",
        "",
        f"- Model: `{model}`",
        f"- Final heldout certification: **{overall_status}**",
        "- Handle selection and hyperparameter fitting: `Dfit/Dcal` only",
        "- Heldout certification: `Dte`, loaded after all handles were frozen",
        "",
        "## Selected handles",
        "",
        "| Variable | Calibration ID | Sites and weights | Strength | Selection evidence |",
        "| --- | --- | --- | ---: | --- |",
        *handle_rows,
        "",
        "## Final Dte certification",
        "",
        "| Directed edge | Result | Causal score | Restoration |",
        "| --- | --- | ---: | --- |",
        *certification_rows,
        "",
        "## Search coverage",
        "",
        f"- Coarse sites ranked: {coarse_count}",
        f"- R-refinement candidates: {r_refinement_count}",
        f"- D-discovery candidates: {d_count}",
        f"- D-refinement candidates: {d_refinement_count}",
        "",
        "## Detailed audit record",
        "",
        (
            "Full rankings, calibration tables, split metrics, restoration tests, "
            f"and manifests are stored in `{detailed_output_path.name}`."
        ),
        "",
    ]
    return "\n".join(lines)


def save_readable_summary(
    output_directory: Path,
    r_discovery: RDiscovery,
    d_discovery: DDiscovery,
    certification: JsonDict,
    certification_passed: bool,
    detailed_output_path: Path,
) -> Path:
    summary_path = output_directory / "automatic_gradual_discovery_summary.md"
    summary = build_readable_summary(
        r_discovery,
        d_discovery,
        certification,
        certification_passed,
        detailed_output_path,
    )
    summary_path.write_text(summary, encoding="utf-8")
    return summary_path


def save_intermediate_results(
    output_directory: Path,
    r_discovery: RDiscovery,
    d_discovery: DDiscovery | None = None,
) -> None:
    if r_discovery.refinement is not None:
        atomic_json(
            output_directory / "R_refinement.json",
            r_discovery.refinement,
        )
    if d_discovery is not None:
        atomic_json(
            output_directory / "D_discovery.json",
            d_discovery.discovery_result,
        )
        if d_discovery.refinement is not None:
            atomic_json(
                output_directory / "D_refinement.json",
                d_discovery.refinement,
            )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    strengths = parse_strengths(args.strength_grid)
    context, encoder, model_info, sparse_records = setup_experiment(args)

    # These functions create model activations for Dfit/Dcal only.
    coarse_dataset = build_coarse_dataset(context, encoder)
    graded_dataset = build_graded_d_dataset(context, encoder)

    # Phase 1-2: discover and refine R without touching Dte.
    r_discovery = discover_r_chain(
        context,
        coarse_dataset,
        graded_dataset,
        strengths,
    )
    save_intermediate_results(args.out_dir, r_discovery)

    # Phase 3-4: discover and refine D without touching Dte.
    d_discovery = discover_d_chain(
        context,
        graded_dataset,
        r_discovery.final,
        strengths,
    )
    save_intermediate_results(args.out_dir, r_discovery, d_discovery)

    # Phase 5: all choices are frozen; Dte becomes available only here.
    certification, certification_passed = certify_final_model(
        context,
        coarse_dataset,
        graded_dataset,
        r_discovery,
        d_discovery,
    )

    result = build_result(
        context,
        model_info,
        sparse_records,
        coarse_dataset,
        graded_dataset,
        r_discovery,
        d_discovery,
        certification,
        certification_passed,
    )
    output_path = args.out_dir / "automatic_gradual_discovery.json"
    atomic_json(output_path, result)
    save_intermediate_results(args.out_dir, r_discovery, d_discovery)
    summary_path = save_readable_summary(
        args.out_dir,
        r_discovery,
        d_discovery,
        certification,
        certification_passed,
        output_path,
    )

    if not certification_passed:
        raise RuntimeError(
            f"Final Dte certification failed; inspect {output_path}"
        )

    print(
        json.dumps(
            {
                "status": "complete",
                "final_model": result["final_model"],
                "output": str(output_path),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()