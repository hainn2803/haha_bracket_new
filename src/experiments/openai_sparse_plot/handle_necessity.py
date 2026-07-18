from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .activation import ChannelSite, find_first_quote_token_position
from .bracket_group_depth_plot import build_unique_discovery_bank
from .runtime import quote_token_ids


PRUNED_HOOK_SUFFIXES: tuple[str, ...] = (
    "attn.act_in",
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.resid_delta",
    "mlp.act_in",
    "mlp.post_act",
    "mlp.resid_delta",
)

QUOTE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("assign", "x = {quote}{content}"),
    ("print", "print({quote}{content}"),
    ("paren_assign", "value = ({quote}{content}"),
    ("handler_arg", "handler(prefix, ({quote}{content}"),
)


@dataclass(frozen=True)
class NecessityExample:
    example_id: str
    prompt: str
    token_ids: tuple[int, ...]
    target_token_id: int
    alt_token_id: int
    split: str
    content_id: str
    stratum: str
    handle_position: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AblationConfiguration:
    config_id: str
    site_ids: tuple[str, ...]
    scope: str
    role: str

    def __post_init__(self) -> None:
        if self.scope not in {"none", "global", "handle_position"}:
            raise ValueError(f"unknown ablation scope: {self.scope}")
        if self.scope == "none" and self.site_ids:
            raise ValueError("clean configuration cannot contain sites")


@dataclass(frozen=True)
class CandidateCircuit:
    csv_path: str
    csv_sha256: str
    sites: tuple[ChannelSite, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(site.site_id for site in self.sites)

    @property
    def retained_by_hook(self) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for site in self.sites:
            grouped[site.hook_key].append(int(site.channel))
        return {hook: tuple(dict.fromkeys(values)) for hook, values in grouped.items()}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_circuit(path: str | Path, *, expected_count: int) -> CandidateCircuit:
    csv_path = Path(path)
    rows = tuple(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != int(expected_count):
        raise ValueError(f"expected {expected_count} candidate rows, found {len(rows)}")
    node_ids = [str(row["node_id"]) for row in rows]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("candidate CSV contains duplicate node IDs")
    return CandidateCircuit(
        csv_path=str(csv_path),
        csv_sha256=file_sha256(csv_path),
        sites=tuple(ChannelSite.from_node_id(node_id) for node_id in node_ids),
        rows=tuple(dict(row) for row in rows),
    )


def pruning_hook_keys(n_layer: int) -> tuple[str, ...]:
    if int(n_layer) <= 0:
        raise ValueError("n_layer must be positive")
    return tuple(
        [f"{layer}.{suffix}" for layer in range(int(n_layer)) for suffix in PRUNED_HOOK_SUFFIXES]
        + ["final_resid"]
    )


def quote_contents(n: int, *, offset: int) -> tuple[str, ...]:
    if int(n) <= 0 or int(offset) < 0:
        raise ValueError("invalid quote content range")
    stems = ("alpha", "beta", "gamma", "delta", "hello", "path", "token", "value")
    return tuple(f"{stems[(offset + i) % len(stems)]} sample {offset + i}" for i in range(int(n)))


def build_quote_necessity_bank(
    enc: Any,
    *,
    fit_contents: int,
    test_contents: int,
    content_offset: int,
) -> tuple[NecessityExample, ...]:
    tokens = quote_token_ids(enc)
    contents = quote_contents(int(fit_contents) + int(test_contents), offset=int(content_offset))
    examples: list[NecessityExample] = []
    for local_index, content in enumerate(contents):
        split = "Dfit" if local_index < int(fit_contents) else "Dte"
        content_id = f"quote-{int(content_offset) + local_index:06d}"
        for template_id, template in QUOTE_TEMPLATES:
            for quote_type, quote in (("single", "'"), ("double", '"')):
                prompt = template.format(quote=quote, content=content)
                token_ids = tuple(int(value) for value in enc.encode(prompt))
                opener = find_first_quote_token_position(enc, token_ids, quote)
                target = int(tokens[quote_type])
                alt = int(tokens["double" if quote_type == "single" else "single"])
                examples.append(
                    NecessityExample(
                        example_id=f"{split}-{content_id}-{template_id}-{quote_type}",
                        prompt=prompt,
                        token_ids=token_ids,
                        target_token_id=target,
                        alt_token_id=alt,
                        split=split,
                        content_id=content_id,
                        stratum=f"{template_id}:{quote_type}",
                        handle_position=int(opener),
                        metadata={"template": template_id, "quote_type": quote_type, "content": content},
                    )
                )
    return tuple(examples)


def build_bracket_necessity_bank(
    enc: Any,
    *,
    fit_contents: int,
    test_contents: int,
    content_offset: int,
) -> tuple[NecessityExample, ...]:
    bank = build_unique_discovery_bank(
        enc,
        contents=int(fit_contents) + int(test_contents),
        fit_contents=int(fit_contents),
        cal_contents=0,
        test_contents=int(test_contents),
        content_offset=int(content_offset),
        depths=(1, 2, 3, 4),
    )
    single = int(enc.encode("]\n")[0])
    double = int(enc.encode("]]\n")[0])
    examples = []
    for row in bank:
        target = double if int(row.close_count) == 2 else single
        alt = single if int(row.close_count) == 2 else double
        examples.append(
            NecessityExample(
                example_id=row.example_id,
                prompt=row.prompt,
                token_ids=tuple(int(value) for value in row.token_ids),
                target_token_id=target,
                alt_token_id=alt,
                split=row.split,
                content_id=str(row.numeric_content),
                stratum=f"d{row.depth}:{row.context_family}",
                handle_position=len(row.token_ids) - 1,
                metadata={
                    "depth": int(row.depth),
                    "close_count": int(row.close_count),
                    "context_family": row.context_family,
                    "numeric_content": row.numeric_content,
                    "surface_open_count": int(row.surface_open_count),
                    "surface_close_count": int(row.surface_close_count),
                },
            )
        )
    return tuple(examples)


def bank_manifest(examples: Sequence[NecessityExample]) -> dict[str, Any]:
    by_split: dict[str, list[NecessityExample]] = defaultdict(list)
    for example in examples:
        by_split[example.split].append(example)
    splits = {}
    for split, rows in sorted(by_split.items()):
        splits[split] = {
            "examples": len(rows),
            "content_ids": sorted({row.content_id for row in rows}),
            "content_count": len({row.content_id for row in rows}),
            "strata": sorted({row.stratum for row in rows}),
        }
    content_sets = [set(value["content_ids"]) for value in splits.values()]
    return {
        "total_examples": len(examples),
        "splits": splits,
        "content_splits_disjoint": all(
            not (left & right)
            for index, left in enumerate(content_sets)
            for right in content_sets[index + 1 :]
        ),
    }


def _group_indices_by_length(examples: Sequence[NecessityExample]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[len(example.token_ids)].append(index)
    return dict(sorted(grouped.items()))


def estimate_hook_means(
    model: Any,
    examples: Sequence[NecessityExample],
    *,
    hook_keys: Sequence[str],
    device: str,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not examples:
        raise ValueError("mean reference bank is empty")
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    interventions: dict[str, Any] = {}
    for hook_key in hook_keys:

        def _accumulate(tensor: torch.Tensor, *, hook_key: str = hook_key) -> torch.Tensor:
            seen.add(hook_key)
            leading_dims = tuple(range(tensor.ndim - 1))
            reduced = tensor.detach().to(torch.float64).sum(dim=leading_dims).cpu()
            sums[hook_key] = reduced if hook_key not in sums else sums[hook_key] + reduced
            counts[hook_key] += int(tensor.numel() // tensor.shape[-1])
            return tensor

        interventions[hook_key] = _accumulate

    for _length, indices in _group_indices_by_length(examples).items():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            token_ids = torch.tensor(
                [examples[index].token_ids for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    model(token_ids)
    missing = set(hook_keys) - seen
    if missing:
        raise RuntimeError(f"mean estimation did not observe hooks: {sorted(missing)}")
    means = {key: (value / float(counts[key])).to(torch.float32) for key, value in sums.items()}
    return means, dict(counts)


def validate_means(
    means: Mapping[str, torch.Tensor],
    *,
    hook_keys: Sequence[str],
    sites: Sequence[ChannelSite],
) -> None:
    missing = set(hook_keys) - set(means)
    if missing:
        raise ValueError(f"missing mean vectors: {sorted(missing)}")
    for site in sites:
        width = int(means[site.hook_key].numel())
        if not 0 <= int(site.channel) < width:
            raise ValueError(f"site outside hook width: {site.site_id} width={width}")


def evaluate_ablation_configs(
    model: Any,
    examples: Sequence[NecessityExample],
    *,
    configs: Sequence[AblationConfiguration],
    sites: Sequence[ChannelSite],
    hook_means: Mapping[str, torch.Tensor],
    hook_keys: Sequence[str],
    retained_by_hook: Mapping[str, Sequence[int]],
    circuit_only: bool,
    device: str,
    max_batch_size: int,
) -> dict[str, np.ndarray]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not configs or not examples:
        raise ValueError("configs and examples must be nonempty")
    site_index = {site.site_id: index for index, site in enumerate(sites)}
    members = []
    for config in configs:
        unknown = set(config.site_ids) - set(site_index)
        if unknown:
            raise ValueError(f"unknown sites in {config.config_id}: {sorted(unknown)}")
        members.append(tuple(site_index[value] for value in config.site_ids))
    config_count = len(configs)
    example_count = len(examples)
    nll = np.empty((config_count, example_count), dtype=np.float32)
    margin = np.empty((config_count, example_count), dtype=np.float32)
    correct = np.empty((config_count, example_count), dtype=np.bool_)
    site_hooks = tuple(site.hook_key for site in sites)
    site_channels = tuple(int(site.channel) for site in sites)
    grouped = _group_indices_by_length(examples)

    for _length, bucket_indices in grouped.items():
        bucket_count = len(bucket_indices)
        bucket_tokens = torch.tensor(
            [examples[index].token_ids for index in bucket_indices],
            dtype=torch.long,
            device=device,
        )
        task_count = config_count * bucket_count
        for task_start in range(0, task_count, int(max_batch_size)):
            task_stop = min(task_count, task_start + int(max_batch_size))
            flat = list(range(task_start, task_stop))
            config_indices = [value // bucket_count for value in flat]
            local_example_indices = [value % bucket_count for value in flat]
            global_example_indices = [bucket_indices[value] for value in local_example_indices]
            token_batch = bucket_tokens[torch.tensor(local_example_indices, dtype=torch.long, device=device)]
            patch_rows_by_hook: dict[str, list[int]] = defaultdict(list)
            patch_channels_by_hook: dict[str, list[int]] = defaultdict(list)
            patch_positions_by_hook: dict[str, list[int | None]] = defaultdict(list)
            for batch_row, (config_index, example_index) in enumerate(zip(config_indices, global_example_indices)):
                config = configs[config_index]
                for member_index in members[config_index]:
                    hook = site_hooks[member_index]
                    patch_rows_by_hook[hook].append(batch_row)
                    patch_channels_by_hook[hook].append(site_channels[member_index])
                    patch_positions_by_hook[hook].append(
                        None if config.scope == "global" else int(examples[example_index].handle_position)
                    )

            active_hooks = tuple(hook_keys) if circuit_only else tuple(sorted(patch_rows_by_hook))
            seen: set[str] = set()
            interventions: dict[str, Any] = {}
            for hook_key in active_hooks:
                rows = tuple(patch_rows_by_hook.get(hook_key, ()))
                channels = tuple(patch_channels_by_hook.get(hook_key, ()))
                positions = tuple(patch_positions_by_hook.get(hook_key, ()))
                retained = tuple(int(value) for value in retained_by_hook.get(hook_key, ()))
                mean_cpu = hook_means[hook_key]

                def _intervene(
                    tensor: torch.Tensor,
                    *,
                    hook_key: str = hook_key,
                    rows: tuple[int, ...] = rows,
                    channels: tuple[int, ...] = channels,
                    positions: tuple[int | None, ...] = positions,
                    retained: tuple[int, ...] = retained,
                    mean_cpu: torch.Tensor = mean_cpu,
                ) -> torch.Tensor:
                    seen.add(hook_key)
                    mean = mean_cpu.to(device=tensor.device, dtype=tensor.dtype)
                    if circuit_only:
                        patched = mean.view(*([1] * (tensor.ndim - 1)), -1).expand_as(tensor).clone()
                        if retained:
                            retained_index = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                            patched.index_copy_(-1, retained_index, tensor.index_select(-1, retained_index))
                    elif rows:
                        patched = tensor.clone()
                    else:
                        return tensor
                    for row, channel, position in zip(rows, channels, positions):
                        if position is None:
                            patched[int(row), ..., int(channel)] = mean[int(channel)]
                        else:
                            patched[int(row), int(position), int(channel)] = mean[int(channel)]
                    return patched

                interventions[hook_key] = _intervene

            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    logits, _, _ = model(token_batch)
            missing = set(active_hooks) - seen
            if missing:
                raise RuntimeError(f"ablation forward did not observe hooks: {sorted(missing)}")
            last = logits[:, -1, :].to(torch.float32)
            targets = torch.tensor(
                [examples[index].target_token_id for index in global_example_indices],
                dtype=torch.long,
                device=last.device,
            )
            alternatives = torch.tensor(
                [examples[index].alt_token_id for index in global_example_indices],
                dtype=torch.long,
                device=last.device,
            )
            rows_tensor = torch.arange(last.shape[0], dtype=torch.long, device=last.device)
            target_logits = last[rows_tensor, targets]
            alt_logits = last[rows_tensor, alternatives]
            batch_nll = F.cross_entropy(last, targets, reduction="none")
            batch_margin = target_logits - alt_logits
            batch_correct = batch_margin > 0
            for batch_row, (config_index, example_index) in enumerate(zip(config_indices, global_example_indices)):
                nll[config_index, example_index] = float(batch_nll[batch_row])
                margin[config_index, example_index] = float(batch_margin[batch_row])
                correct[config_index, example_index] = bool(batch_correct[batch_row])
    return {"nll": nll, "margin": margin, "correct": correct}


def _bootstrap_interval(cluster_values: np.ndarray, *, repetitions: int, seed: int) -> tuple[float, float]:
    if cluster_values.ndim != 1 or not len(cluster_values):
        raise ValueError("bootstrap values must be a nonempty vector")
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(cluster_values), size=(int(repetitions), len(cluster_values)))
    means = cluster_values[sampled].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_ablation_results(
    examples: Sequence[NecessityExample],
    configs: Sequence[AblationConfiguration],
    results: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    content_ids = tuple(sorted({example.content_id for example in examples}))
    content_indices = {
        content_id: np.asarray([index for index, row in enumerate(examples) if row.content_id == content_id], dtype=int)
        for content_id in content_ids
    }
    baseline_nll = np.asarray(baseline["nll"])[0]
    baseline_margin = np.asarray(baseline["margin"])[0]
    baseline_contrast_loss = np.logaddexp(0.0, -baseline_margin)
    baseline_correct = np.asarray(baseline["correct"])[0].astype(np.float32)
    summaries = []
    for config_index, config in enumerate(configs):
        current_nll = np.asarray(results["nll"])[config_index]
        current_margin = np.asarray(results["margin"])[config_index]
        current_contrast_loss = np.logaddexp(0.0, -current_margin)
        current_correct = np.asarray(results["correct"])[config_index].astype(np.float32)
        nll_delta = current_nll - baseline_nll
        contrast_loss_delta = current_contrast_loss - baseline_contrast_loss
        margin_drop = baseline_margin - current_margin
        accuracy_drop = baseline_correct - current_correct
        cluster_nll = np.asarray([nll_delta[indices].mean() for indices in content_indices.values()])
        cluster_contrast_loss = np.asarray(
            [contrast_loss_delta[indices].mean() for indices in content_indices.values()]
        )
        cluster_margin = np.asarray([margin_drop[indices].mean() for indices in content_indices.values()])
        cluster_accuracy = np.asarray([accuracy_drop[indices].mean() for indices in content_indices.values()])
        strata = {}
        for stratum in sorted({example.stratum for example in examples}):
            indices = np.asarray([index for index, row in enumerate(examples) if row.stratum == stratum], dtype=int)
            strata[stratum] = {
                "examples": int(len(indices)),
                "mean_nll_delta": float(nll_delta[indices].mean()),
                "mean_contrast_loss_delta": float(contrast_loss_delta[indices].mean()),
                "mean_margin_drop": float(margin_drop[indices].mean()),
                "baseline_accuracy": float(baseline_correct[indices].mean()),
                "ablated_accuracy": float(current_correct[indices].mean()),
            }
        summaries.append(
            {
                "config_id": config.config_id,
                "site_ids": list(config.site_ids),
                "scope": config.scope,
                "role": config.role,
                "examples": len(examples),
                "content_clusters": len(content_ids),
                "baseline_mean_nll": float(baseline_nll.mean()),
                "ablated_mean_nll": float(current_nll.mean()),
                "mean_nll_delta": float(nll_delta.mean()),
                "nll_delta_ci95": list(
                    _bootstrap_interval(
                        cluster_nll,
                        repetitions=int(bootstrap_repetitions),
                        seed=int(bootstrap_seed) + config_index * 17,
                    )
                ),
                "baseline_mean_signed_margin": float(baseline_margin.mean()),
                "ablated_mean_signed_margin": float(current_margin.mean()),
                "baseline_mean_contrast_loss": float(baseline_contrast_loss.mean()),
                "ablated_mean_contrast_loss": float(current_contrast_loss.mean()),
                "mean_contrast_loss_delta": float(contrast_loss_delta.mean()),
                "contrast_loss_delta_ci95": list(
                    _bootstrap_interval(
                        cluster_contrast_loss,
                        repetitions=int(bootstrap_repetitions),
                        seed=int(bootstrap_seed) + config_index * 17 + 3,
                    )
                ),
                "mean_margin_drop": float(margin_drop.mean()),
                "margin_drop_ci95": list(
                    _bootstrap_interval(
                        cluster_margin,
                        repetitions=int(bootstrap_repetitions),
                        seed=int(bootstrap_seed) + config_index * 17 + 1,
                    )
                ),
                "baseline_contrast_accuracy": float(baseline_correct.mean()),
                "ablated_contrast_accuracy": float(current_correct.mean()),
                "mean_accuracy_drop": float(accuracy_drop.mean()),
                "accuracy_drop_ci95": list(
                    _bootstrap_interval(
                        cluster_accuracy,
                        repetitions=int(bootstrap_repetitions),
                        seed=int(bootstrap_seed) + config_index * 17 + 2,
                    )
                ),
                "clear_positive_nll_effect": bool(
                    _bootstrap_interval(
                        cluster_nll,
                        repetitions=int(bootstrap_repetitions),
                        seed=int(bootstrap_seed) + config_index * 17,
                    )[0]
                    > 0.0
                ),
                "strata": strata,
            }
        )
    return summaries


def singleton_ranks(
    summaries: Sequence[Mapping[str, Any]],
    *,
    metric: str = "mean_nll_delta",
) -> dict[str, int]:
    singles = [row for row in summaries if str(row["config_id"]).startswith("single::")]
    if metric not in {
        "mean_nll_delta",
        "mean_contrast_loss_delta",
        "mean_margin_drop",
        "mean_accuracy_drop",
    }:
        raise ValueError(f"unsupported singleton ranking metric: {metric}")
    ranked = sorted(singles, key=lambda row: (-float(row[metric]), str(row["config_id"])))
    return {str(row["site_ids"][0]): index + 1 for index, row in enumerate(ranked)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def save_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def chunked(values: Sequence[Any], size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    for start in range(0, len(values), int(size)):
        yield start, values[start : start + int(size)]
