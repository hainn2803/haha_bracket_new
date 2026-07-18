from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ablate_rediscover import (
    ClampedRun,
    HandleConfiguration,
    RediscoveryExample,
    RediscoveryPair,
)
from .activation import ChannelSite


@dataclass(frozen=True)
class BinaryScalarReadout:
    threshold: float
    orientation: int
    fit_accuracy: float

    def predict(self, value: float) -> int:
        signed = float(self.orientation) * (float(value) - float(self.threshold))
        return 1 if signed > 0.0 else -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "orientation": int(self.orientation),
            "fit_accuracy": float(self.fit_accuracy),
        }


def fit_binary_scalar_readout(values: Sequence[float], labels: Sequence[int]) -> BinaryScalarReadout:
    if len(values) != len(labels) or not values:
        raise ValueError("readout values and labels must be nonempty and aligned")
    unique = sorted({float(value) for value in values})
    thresholds = [unique[0] - 1e-6, unique[-1] + 1e-6]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    best: tuple[float, int, float] | None = None
    for orientation in (-1, 1):
        for threshold in thresholds:
            predicted = [1 if orientation * (float(value) - threshold) > 0.0 else -1 for value in values]
            accuracy = float(np.mean([pred == int(label) for pred, label in zip(predicted, labels)]))
            key = (accuracy, -abs(float(threshold)), orientation)
            if best is None or key > (best[0], -abs(best[2]), best[1]):
                best = (accuracy, orientation, float(threshold))
    assert best is not None
    return BinaryScalarReadout(threshold=best[2], orientation=best[1], fit_accuracy=best[0])


def readout_accuracy(
    readout: BinaryScalarReadout,
    examples: Sequence[RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    *,
    split: str,
    site_id: str,
) -> float:
    rows = [row for row in examples if row.split == split]
    return float(
        np.mean(
            [
                readout.predict(runs[row.example_id].features_by_site[site_id]) == row.variable_value
                for row in rows
            ]
        )
    )


def evaluate_progressive_configurations(
    model: Any,
    configs: Sequence[HandleConfiguration],
    pairs: Sequence[RediscoveryPair],
    *,
    examples: Mapping[str, RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    site_lookup: Mapping[str, ChannelSite],
    probe_sites: Sequence[ChannelSite],
    negative_token_id: int,
    positive_token_id: int,
    device: str,
    max_batch_size: int,
    restore_probes_to_base: bool = False,
    restore_probe_site_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not configs or not pairs or not probe_sites:
        raise ValueError("configs, pairs, and probe sites must be nonempty")
    margins = np.empty((len(configs), len(pairs)), dtype=np.float32)
    probes = np.empty((len(configs), len(pairs), len(probe_sites)), dtype=np.float32)
    pairs_by_length: dict[int, list[int]] = defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        pairs_by_length[len(examples[pair.base_id].token_ids)].append(pair_index)
    probes_by_hook: dict[str, list[tuple[int, ChannelSite]]] = defaultdict(list)
    for probe_index, site in enumerate(probe_sites):
        probes_by_hook[site.hook_key].append((probe_index, site))
    restore_ids = (
        {site.site_id for site in probe_sites}
        if restore_probes_to_base
        else {str(site_id) for site_id in (restore_probe_site_ids or ())}
    )
    for pair_indices in pairs_by_length.values():
        task_count = len(configs) * len(pair_indices)
        for task_start in range(0, task_count, int(max_batch_size)):
            flat = list(range(task_start, min(task_count, task_start + int(max_batch_size))))
            config_indices = [value // len(pair_indices) for value in flat]
            local_pair_indices = [value % len(pair_indices) for value in flat]
            global_pair_indices = [pair_indices[value] for value in local_pair_indices]
            base_examples = [examples[pairs[index].base_id] for index in global_pair_indices]
            token_ids = torch.tensor([row.token_ids for row in base_examples], dtype=torch.long, device=device)
            positions = torch.tensor([row.patch_position for row in base_examples], dtype=torch.long, device=device)
            batch_rows = torch.arange(len(flat), dtype=torch.long, device=device)
            patches_by_hook: dict[str, list[tuple[int, int, int, float, float]]] = defaultdict(list)
            for batch_row, (config_index, pair_index) in enumerate(zip(config_indices, global_pair_indices)):
                config = configs[config_index]
                pair = pairs[pair_index]
                source = runs[pair.source_id]
                position = int(examples[pair.base_id].patch_position)
                for site_id, weight in config.weights_by_site.items():
                    site = site_lookup[site_id]
                    patches_by_hook[site.hook_key].append(
                        (
                            batch_row,
                            position,
                            int(site.channel),
                            float(source.features_by_site[site_id]),
                            float(config.strength) * float(weight),
                        )
                    )
            active_hooks = sorted(set(patches_by_hook) | set(probes_by_hook))
            recorded: dict[int, torch.Tensor] = {}
            interventions: dict[str, Any] = {}
            for hook_key in active_hooks:
                patches = tuple(patches_by_hook.get(hook_key, ()))
                hook_probes = tuple(probes_by_hook.get(hook_key, ()))

                def _intervene(
                    tensor: torch.Tensor,
                    *,
                    patches: tuple[tuple[int, int, int, float, float], ...] = patches,
                    hook_probes: tuple[tuple[int, ChannelSite], ...] = hook_probes,
                ) -> torch.Tensor:
                    patched = tensor.clone() if patches or any(site.site_id in restore_ids for _, site in hook_probes) else tensor
                    for row, position, channel, source_value, coefficient in patches:
                        current = patched[int(row), int(position), int(channel)]
                        source_value_tensor = torch.as_tensor(
                            source_value, device=tensor.device, dtype=tensor.dtype
                        )
                        patched[int(row), int(position), int(channel)] = current + float(coefficient) * (
                            source_value_tensor - current
                        )
                    if restore_ids:
                        for _probe_index, site in hook_probes:
                            if site.site_id not in restore_ids:
                                continue
                            base_values = torch.tensor(
                                [
                                    runs[pairs[pair_index].base_id].features_by_site[site.site_id]
                                    for pair_index in global_pair_indices
                                ],
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            patched[batch_rows, positions, int(site.channel)] = base_values
                    for probe_index, site in hook_probes:
                        recorded[probe_index] = patched[
                            batch_rows, positions, int(site.channel)
                        ].detach().cpu()
                    return patched

                interventions[hook_key] = _intervene
            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    logits, _, _ = model(token_ids)
            last = logits[:, -1, :].to(torch.float32)
            batch_margins = (
                last[:, int(positive_token_id)] - last[:, int(negative_token_id)]
            ).detach().cpu().numpy()
            if set(recorded) != set(range(len(probe_sites))):
                raise RuntimeError("not all progressive probe sites were recorded")
            for batch_row, (config_index, pair_index) in enumerate(zip(config_indices, global_pair_indices)):
                margins[config_index, pair_index] = float(batch_margins[batch_row])
                for probe_index in range(len(probe_sites)):
                    probes[config_index, pair_index, probe_index] = float(recorded[probe_index][batch_row])
    return margins, probes


def progressive_abstract_signature(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    *,
    components: int = 2,
) -> tuple[float, ...]:
    values: list[float] = []
    for pair in pairs:
        delta = float(examples[pair.source_id].variable_value - examples[pair.base_id].variable_value)
        values.extend([delta] * int(components))
    return tuple(values)


def progressive_neural_signature(
    pairs: Sequence[RediscoveryPair],
    runs: Mapping[str, ClampedRun],
    patched_margins: Sequence[float],
    patched_probes: np.ndarray,
    *,
    probe_site_ids: Sequence[str],
    probe_scales: Sequence[float] | None = None,
    include_output: bool = True,
) -> tuple[float, ...]:
    if patched_probes.shape != (len(pairs), len(probe_site_ids)):
        raise ValueError("progressive probe array shape mismatch")
    scales = tuple(float(value) for value in (probe_scales or (1.0,) * len(probe_site_ids)))
    if len(scales) != len(probe_site_ids):
        raise ValueError("probe scale count must match probe sites")
    values: list[float] = []
    for pair_index, pair in enumerate(pairs):
        base = runs[pair.base_id]
        for probe_index, site_id in enumerate(probe_site_ids):
            values.append(
                scales[probe_index]
                * float(patched_probes[pair_index, probe_index] - base.features_by_site[site_id])
            )
        if include_output:
            values.append(float(patched_margins[pair_index] - base.class_margin))
    return tuple(values)


def moves_toward_source(base: float, source: float, patched: float) -> bool:
    if abs(float(source) - float(base)) <= 1e-8:
        return abs(float(patched) - float(base)) <= 1e-6
    return abs(float(source) - float(patched)) < abs(float(source) - float(base))


def progressive_relation_summary(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    patched_margins: Sequence[float],
    patched_probes: np.ndarray,
    *,
    probe_site_id: str,
    readout: BinaryScalarReadout,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, bool]]] = defaultdict(list)
    sensitivity_rows: list[dict[str, bool]] = []
    invariance_rows: list[dict[str, bool]] = []
    for index, pair in enumerate(pairs):
        base_ex = examples[pair.base_id]
        source_ex = examples[pair.source_id]
        base = runs[pair.base_id]
        source = runs[pair.source_id]
        expected = source_ex.variable_value if source_ex.variable_value != base_ex.variable_value else base_ex.variable_value
        output_prediction = 1 if float(patched_margins[index]) > 0.0 else -1
        probe_value = float(patched_probes[index, 0])
        record = {
            "output_correct": output_prediction == expected,
            "probe_state_correct": readout.predict(probe_value) == expected,
            "probe_moves_toward_source": moves_toward_source(
                    base.features_by_site[probe_site_id],
                    source.features_by_site[probe_site_id],
                    probe_value,
            ),
        }
        rows[pair.relation].append(record)
        (sensitivity_rows if source_ex.variable_value != base_ex.variable_value else invariance_rows).append(record)
    relation_metrics: dict[str, Any] = {}
    acceptance_values: list[float] = []
    for relation, relation_rows in sorted(rows.items()):
        metrics = {
            key: float(np.mean([row[key] for row in relation_rows]))
            for key in ("output_correct", "probe_state_correct", "probe_moves_toward_source")
        }
        relation_metrics[relation] = metrics
        acceptance_values.extend((metrics["output_correct"], metrics["probe_state_correct"]))
    balanced_blocks = {
        "sensitivity_output": float(np.mean([row["output_correct"] for row in sensitivity_rows])),
        "sensitivity_Rmid_state": float(np.mean([row["probe_state_correct"] for row in sensitivity_rows])),
        "invariance_output": float(np.mean([row["output_correct"] for row in invariance_rows])),
        "invariance_Rmid_state": float(np.mean([row["probe_state_correct"] for row in invariance_rows])),
    }
    return {
        "relations": relation_metrics,
        "balanced_blocks": balanced_blocks,
        "score": float(np.mean(list(balanced_blocks.values()))),
        "all_required_rates_at_least_0_90": bool(
            acceptance_values and min(acceptance_values) >= 0.90
        ),
    }


def mediation_summary(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    runs: Mapping[str, ClampedRun],
    direct_margins: Sequence[float],
    restored_margins: Sequence[float],
) -> dict[str, Any]:
    sensitivity_indices = [
        index
        for index, pair in enumerate(pairs)
        if examples[pair.base_id].variable_value != examples[pair.source_id].variable_value
    ]
    if not sensitivity_indices:
        raise ValueError("mediation requires different-R pairs")
    direct_source = []
    restored_base = []
    removed_fractions = []
    for index in sensitivity_indices:
        pair = pairs[index]
        base_value = examples[pair.base_id].variable_value
        source_value = examples[pair.source_id].variable_value
        direct_prediction = 1 if float(direct_margins[index]) > 0.0 else -1
        restored_prediction = 1 if float(restored_margins[index]) > 0.0 else -1
        direct_source.append(direct_prediction == source_value)
        restored_base.append(restored_prediction == base_value)
        base_margin = float(runs[pair.base_id].class_margin)
        direct_effect = abs(float(direct_margins[index]) - base_margin)
        residual = abs(float(restored_margins[index]) - base_margin)
        if direct_effect > 1e-8:
            removed_fractions.append(1.0 - residual / direct_effect)
    return {
        "different_R_records": len(sensitivity_indices),
        "direct_output_matches_source": float(np.mean(direct_source)),
        "restored_Rmid_output_preserves_base": float(np.mean(restored_base)),
        "mean_output_effect_removed_fraction": float(np.mean(removed_fractions)),
        "passes": bool(
            np.mean(direct_source) >= 0.90
            and np.mean(restored_base) >= 0.90
            and np.mean(removed_fractions) >= 0.50
        ),
    }


def append_signature(path: Path, site_id: str, signature: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"site_id": site_id, "signature": list(signature)}, sort_keys=True) + "\n")


def load_signatures(path: Path) -> dict[str, tuple[float, ...]]:
    if not path.exists():
        return {}
    output: dict[str, tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        output[str(row["site_id"])] = tuple(float(value) for value in row["signature"])
    return output
