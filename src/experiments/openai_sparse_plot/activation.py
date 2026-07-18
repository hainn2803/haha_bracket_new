from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .schema import SparseCircuitSite


NODE_ID_RE = re.compile(r"^(?P<hook>[^:]+):(?P<channel>-?\d+)$")


@dataclass(frozen=True)
class ChannelSite:
    site_id: str
    hook_key: str
    channel: int
    label: str | None = None

    @classmethod
    def from_node_id(cls, node_id: str, *, label: str | None = None) -> "ChannelSite":
        match = NODE_ID_RE.match(node_id)
        if not match:
            raise ValueError(f"node_id is not a scalar channel site: {node_id}")
        return cls(
            site_id=node_id,
            hook_key=match.group("hook"),
            channel=int(match.group("channel")),
            label=label,
        )

    @classmethod
    def from_sparse_site(cls, site: SparseCircuitSite) -> "ChannelSite":
        if len(site.node_ids) != 1:
            raise ValueError("ChannelSite only supports single-node sites")
        return cls.from_node_id(site.node_ids[0], label=site.label)


def encode_prompt(enc: Any, prompt: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)


def find_first_quote_token_position(enc: Any, token_ids: Sequence[int], quote: str) -> int:
    for idx, tok in enumerate(token_ids):
        if quote in enc.decode([int(tok)]):
            return idx
    raise ValueError(f"quote token {quote!r} not found in token ids")


def quote_positions_for_prompt(enc: Any, prompt: str) -> dict[str, int]:
    token_ids = enc.encode(prompt)
    if '"' in prompt:
        opening = find_first_quote_token_position(enc, token_ids, '"')
        quote_type = "double"
    elif "'" in prompt:
        opening = find_first_quote_token_position(enc, token_ids, "'")
        quote_type = "single"
    else:
        raise ValueError("prompt has no single or double quote")
    return {
        "opening_quote_position": opening,
        "final_position": len(token_ids) - 1,
        "quote_type": quote_type,
    }


def binary_quote_margin(logits: torch.Tensor, *, single_token_id: int, double_token_id: int) -> float:
    last = logits[0, -1]
    return float(last[double_token_id] - last[single_token_id])


def _site_regex(sites: Sequence[ChannelSite]) -> str:
    escaped = [re.escape(site.hook_key) for site in sites]
    return "^(?:" + "|".join(sorted(set(escaped))) + ")$"


def record_activations(
    model: Any,
    token_ids: torch.Tensor,
    sites: Sequence[ChannelSite],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    regex = _site_regex(sites)
    with torch.no_grad():
        with hook_recorder(regex=regex) as ctx:
            logits, _, _ = model(token_ids)
    return logits, {k: v.detach().cpu() for k, v in ctx.items()}


def extract_site_values(
    activation_cache: Mapping[str, torch.Tensor],
    sites: Sequence[ChannelSite],
    *,
    positions: Sequence[int],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for site in sites:
        tensor = activation_cache[site.hook_key]
        values = []
        for pos in positions:
            values.append(float(tensor[0, pos, site.channel]))
        out[site.site_id] = values
    return out


def make_channel_patch(
    site: ChannelSite,
    *,
    source_cache: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    source_positions: Sequence[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    source_positions = tuple(source_positions if source_positions is not None else positions)
    positions = tuple(positions)
    if len(positions) != len(source_positions):
        raise ValueError("positions and source_positions must have the same length")
    source_tensor = source_cache[site.hook_key]
    source_values = source_tensor[0, list(source_positions), site.channel].detach()

    def _patch(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        values = source_values.to(device=patched.device, dtype=patched.dtype)
        patched[0, list(positions), site.channel] = values
        return patched

    return _patch


def make_multi_channel_patch(
    sites: Sequence[ChannelSite],
    *,
    source_cache: Mapping[str, torch.Tensor],
    positions_by_site: Mapping[str, Sequence[int]],
    source_positions_by_site: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Create hook interventions for a group of scalar channel sites."""

    source_positions_by_site = source_positions_by_site or positions_by_site
    by_hook: dict[str, list[ChannelSite]] = {}
    for site in sites:
        by_hook.setdefault(site.hook_key, []).append(site)

    interventions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    for hook_key, hook_sites in by_hook.items():
        source_tensor = source_cache[hook_key]
        patch_specs = []
        for site in hook_sites:
            positions = tuple(int(x) for x in positions_by_site[site.site_id])
            source_positions = tuple(int(x) for x in source_positions_by_site[site.site_id])
            if len(positions) != len(source_positions):
                raise ValueError("positions and source_positions must have the same length")
            source_values = source_tensor[0, list(source_positions), site.channel].detach()
            patch_specs.append((site.channel, positions, source_values))

        def _patch(tensor: torch.Tensor, *, patch_specs=tuple(patch_specs)) -> torch.Tensor:
            patched = tensor.clone()
            for channel, positions, source_values in patch_specs:
                values = source_values.to(device=patched.device, dtype=patched.dtype)
                patched[0, list(positions), int(channel)] = values
            return patched

        interventions[hook_key] = _patch
    return interventions


def make_weighted_multi_channel_patch(
    sites: Sequence[ChannelSite],
    *,
    source_cache: Mapping[str, torch.Tensor],
    positions_by_site: Mapping[str, Sequence[int]],
    weights_by_site: Mapping[str, float],
    strength: float = 1.0,
    source_positions_by_site: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Create soft PLOT interventions for scalar channel sites.

    For each requested scalar site this applies

        base + strength * weight * (source - base)

    at the requested token position. This is the top-K soft-handle intervention
    used by the PLOT paper, specialized to scalar activation sites.
    """

    source_positions_by_site = source_positions_by_site or positions_by_site
    by_hook: dict[str, list[ChannelSite]] = {}
    for site in sites:
        by_hook.setdefault(site.hook_key, []).append(site)

    interventions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    for hook_key, hook_sites in by_hook.items():
        source_tensor = source_cache[hook_key]
        patch_specs = []
        for site in hook_sites:
            positions = tuple(int(x) for x in positions_by_site[site.site_id])
            source_positions = tuple(int(x) for x in source_positions_by_site[site.site_id])
            if len(positions) != len(source_positions):
                raise ValueError("positions and source_positions must have the same length")
            source_values = source_tensor[0, list(source_positions), site.channel].detach()
            weight = float(weights_by_site.get(site.site_id, 0.0))
            patch_specs.append((site.channel, positions, source_values, weight))

        def _patch(tensor: torch.Tensor, *, patch_specs=tuple(patch_specs)) -> torch.Tensor:
            patched = tensor.clone()
            alpha = float(strength)
            for channel, positions, source_values, weight in patch_specs:
                values = source_values.to(device=patched.device, dtype=patched.dtype)
                idx = list(positions)
                base_values = patched[0, idx, int(channel)]
                patched[0, idx, int(channel)] = base_values + alpha * float(weight) * (values - base_values)
            return patched

        interventions[hook_key] = _patch
    return interventions


def run_with_patch(
    model: Any,
    token_ids: torch.Tensor,
    *,
    site: ChannelSite,
    source_cache: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    source_positions: Sequence[int] | None = None,
) -> torch.Tensor:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = {
        site.hook_key: make_channel_patch(
            site,
            source_cache=source_cache,
            positions=positions,
            source_positions=source_positions,
        )
    }
    with torch.no_grad():
        with hook_recorder(regex="^$", interventions=interventions):
            logits, _, _ = model(token_ids)
    return logits


def run_with_group_patch(
    model: Any,
    token_ids: torch.Tensor,
    *,
    sites: Sequence[ChannelSite],
    source_cache: Mapping[str, torch.Tensor],
    positions_by_site: Mapping[str, Sequence[int]],
    source_positions_by_site: Mapping[str, Sequence[int]] | None = None,
) -> torch.Tensor:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = make_multi_channel_patch(
        sites,
        source_cache=source_cache,
        positions_by_site=positions_by_site,
        source_positions_by_site=source_positions_by_site,
    )
    with torch.no_grad():
        with hook_recorder(regex="^$", interventions=interventions):
            logits, _, _ = model(token_ids)
    return logits


def run_with_weighted_group_patch(
    model: Any,
    token_ids: torch.Tensor,
    *,
    sites: Sequence[ChannelSite],
    source_cache: Mapping[str, torch.Tensor],
    positions_by_site: Mapping[str, Sequence[int]],
    weights_by_site: Mapping[str, float],
    strength: float = 1.0,
    source_positions_by_site: Mapping[str, Sequence[int]] | None = None,
) -> torch.Tensor:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = make_weighted_multi_channel_patch(
        sites,
        source_cache=source_cache,
        positions_by_site=positions_by_site,
        source_positions_by_site=source_positions_by_site,
        weights_by_site=weights_by_site,
        strength=strength,
    )
    with torch.no_grad():
        with hook_recorder(regex="^$", interventions=interventions):
            logits, _, _ = model(token_ids)
    return logits
