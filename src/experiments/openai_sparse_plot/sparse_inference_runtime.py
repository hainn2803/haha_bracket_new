from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class SparseInferenceLinear(nn.Module):
    """Inference-only linear layer with the exact dense weights stored as CSR."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.weight.requires_grad and torch.is_grad_enabled():
            # Conversion is intended for eval/no-grad experiments. The check is
            # informational: parameters commonly still have requires_grad=True.
            pass
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.register_buffer("weight_csr", linear.weight.detach().to_sparse_csr())
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if int(inputs.shape[-1]) != self.in_features:
            raise ValueError(
                f"expected final dimension {self.in_features}, got {int(inputs.shape[-1])}"
            )
        shape = tuple(int(value) for value in inputs.shape[:-1])
        flat = inputs.reshape(-1, self.in_features)
        output = torch.sparse.mm(self.weight_csr, flat.transpose(0, 1)).transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        else:
            output = output.contiguous()
        return output.reshape(*shape, self.out_features)


@dataclass(frozen=True)
class SparseConversionRecord:
    module_name: str
    shape: tuple[int, int]
    nonzero: int
    total: int

    def to_json(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "shape": list(self.shape),
            "nonzero": self.nonzero,
            "total": self.total,
            "zero_fraction": 1.0 - float(self.nonzero) / float(self.total),
        }


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def convert_transformer_linears_to_sparse(
    model: nn.Module,
    *,
    include_names: Sequence[str] = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"),
) -> tuple[SparseConversionRecord, ...]:
    """Replace every declared transformer linear with its exact CSR equivalent.

    The language-model head is intentionally left dense because the bracket
    runner reads only two output rows directly and never forms full logits.
    """

    selected: list[tuple[str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(module_name.endswith(suffix) for suffix in include_names):
            selected.append((module_name, module))
    if not selected:
        raise ValueError("no transformer linear modules matched the sparse conversion policy")

    records: list[SparseConversionRecord] = []
    for module_name, module in selected:
        weight = module.weight.detach()
        record = SparseConversionRecord(
            module_name=module_name,
            shape=(int(weight.shape[0]), int(weight.shape[1])),
            nonzero=int(torch.count_nonzero(weight).item()),
            total=int(weight.numel()),
        )
        parent, attribute = _resolve_parent(model, module_name)
        setattr(parent, attribute, SparseInferenceLinear(module))
        records.append(record)
    return tuple(records)


@dataclass(frozen=True)
class PrefixKVCache:
    token_ids: torch.Tensor
    queries: tuple[torch.Tensor, ...]
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]


def _selected_output_margin(
    model: Any,
    final_hidden: torch.Tensor,
    final_token_ids: torch.Tensor,
    *,
    single_close_token_id: int,
    double_close_token_id: int,
) -> torch.Tensor:
    output_rows = model.lm_head.weight[
        torch.tensor(
            [int(single_close_token_id), int(double_close_token_id)],
            dtype=torch.long,
            device=final_hidden.device,
        )
    ]
    selected_logits = final_hidden @ output_rows.transpose(0, 1)
    selected_logits = selected_logits + model.final_logits_bias[
        [int(single_close_token_id), int(double_close_token_id)]
    ]
    if model.config.enable_bigram_table:
        selected_logits = selected_logits + model.bigram_table[final_token_ids][:, [
            int(single_close_token_id),
            int(double_close_token_id),
        ]]
    return selected_logits[:, 1] - selected_logits[:, 0]


def bracket_forward_without_full_logits(
    model: Any,
    token_ids: torch.Tensor,
    *,
    single_close_token_id: int,
    double_close_token_id: int,
    include_margin: bool,
) -> torch.Tensor | None:
    """Run the exact transformer but avoid constructing every vocabulary logit."""

    from circuit_sparsity.inference.hook_utils import hook_namespace, hook_save

    _batch, length = token_ids.shape
    if int(length) > int(model.config.block_size):
        raise ValueError("input exceeds model block size")
    token_embeddings = model.transformer.wte(token_ids)
    position_embeddings = model.transformer.wpe.weight[:length].unsqueeze(0)
    if model.config.cat_pos_emb:
        hidden = model.transformer.drop(token_embeddings)
        position_to_cat = model.transformer.drop(position_embeddings) if model.config.dropout_cat_pos_emb else position_embeddings
    else:
        hidden = model.transformer.drop(token_embeddings + position_embeddings)
        position_to_cat = None

    for layer_index, block in enumerate(model.transformer.h):
        with hook_namespace(str(layer_index)):
            if model.config.cat_pos_emb:
                hidden = block(hidden, position_to_cat)
            else:
                hidden = block(hidden)
    hidden = hook_save("final_resid", hidden)
    if not include_margin:
        return None

    final_hidden = model.transformer.ln_f(hidden[:, -1, :])
    return _selected_output_margin(
        model,
        final_hidden,
        token_ids[:, -1],
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )


def build_prefix_kv_cache(
    model: Any,
    token_ids: torch.Tensor,
    *,
    single_close_token_id: int,
    double_close_token_id: int,
) -> PrefixKVCache:
    """Cache the unaffected prefix K/V tensors for one clean base prompt."""

    from circuit_sparsity.inference.hook_utils import hook_recorder

    if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
        raise ValueError("prefix cache expects one base prompt")
    layer_count = len(model.transformer.h)
    regex = r"^(?:" + "|".join(str(index) for index in range(layer_count)) + r")\.attn\.(?:q|k|v)$"
    with torch.no_grad():
        with hook_recorder(regex=regex) as context:
            bracket_forward_without_full_logits(
                model,
                token_ids,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
                include_margin=False,
            )
    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.transformer.h):
        for name, output in (("q", queries), ("k", keys), ("v", values)):
            tensor = context[f"{layer_index}.attn.{name}"].detach()
            shaped = tensor.view(1, tensor.shape[1], block.attn.n_head, block.attn.d_head).transpose(1, 2)
            output.append(shaped[:, :, :-1, :].contiguous())
    return PrefixKVCache(
        token_ids=token_ids.detach(),
        queries=tuple(queries),
        keys=tuple(keys),
        values=tuple(values),
    )


def build_prefix_cache_bank(
    model: Any,
    token_ids_by_base: Mapping[str, torch.Tensor],
    *,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, PrefixKVCache]:
    return {
        str(base_id): build_prefix_kv_cache(
            model,
            token_ids,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        for base_id, token_ids in token_ids_by_base.items()
    }


def _final_query_attention(
    attention: Any,
    prefix_q: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    final_q: torch.Tensor,
    final_k: torch.Tensor,
    final_v: torch.Tensor,
) -> torch.Tensor:
    del prefix_q
    if attention.training and float(attention.dropout) != 0.0:
        raise ValueError("incremental inference requires eval mode or zero attention dropout")
    scale = 1.0 / math.sqrt(final_k.size(-1))
    prefix_scores = (final_q * prefix_k).sum(dim=-1) * scale
    final_scores = (final_q * final_k).sum(dim=-1) * scale
    attention_impl = getattr(attention, "attn_imp", None)
    sink_logit = getattr(attention_impl, "sink_logit", None)
    if sink_logit is None:
        scores = torch.cat((prefix_scores, final_scores), dim=-1)
        weights = torch.softmax(scores, dim=-1)
        prefix_weights = weights[..., :-1]
        final_weights = weights[..., -1:]
    else:
        sink_scores = sink_logit.to(dtype=final_q.dtype, device=final_q.device).view(1, -1, 1)
        sink_scores = sink_scores.expand(final_q.shape[0], -1, -1)
        scores = torch.cat((sink_scores, prefix_scores, final_scores), dim=-1)
        weights = torch.softmax(scores, dim=-1)
        prefix_weights = weights[..., 1:-1]
        final_weights = weights[..., -1:]
    prefix_output = (prefix_weights.unsqueeze(-1) * prefix_v).sum(dim=2)
    final_output = final_weights * final_v[:, :, 0, :]
    return (prefix_output + final_output).unsqueeze(2)


def bracket_incremental_final_forward(
    model: Any,
    cache: PrefixKVCache,
    *,
    batch_size: int,
    single_close_token_id: int,
    double_close_token_id: int,
    include_margin: bool,
) -> torch.Tensor | None:
    """Recompute only the final token, using exact clean prefix K/V tensors.

    Final-position interventions cannot affect earlier positions in a causal
    transformer. This function executes the same final-token equations and the
    same hook names while reusing those causally fixed prefix tensors.
    """

    from circuit_sparsity.inference.hook_utils import hook_namespace, hook_save

    token_ids = cache.token_ids
    length = int(token_ids.shape[1])
    final_token = token_ids[:, -1].expand(int(batch_size))
    token_embedding = model.transformer.wte(final_token).unsqueeze(1)
    position_embedding = model.transformer.wpe.weight[length - 1].view(1, 1, -1)
    if model.config.cat_pos_emb:
        hidden = model.transformer.drop(token_embedding)
        position_to_cat = model.transformer.drop(position_embedding) if model.config.dropout_cat_pos_emb else position_embedding
        position_to_cat = position_to_cat.expand(int(batch_size), -1, -1)
    else:
        hidden = model.transformer.drop(token_embedding + position_embedding)
        position_to_cat = None

    for layer_index, block in enumerate(model.transformer.h):
        with hook_namespace(str(layer_index)):
            hidden = hook_save("resid_in", hidden)
            with hook_namespace("attn"):
                attn_input = block.ln_1(hidden)
                if model.config.cat_pos_emb:
                    attn_input = torch.cat((attn_input, block.ln_p1(position_to_cat)), dim=-1)
                attn_input = block.config.maybe_activation_sparsity(attn_input, "attn_in")
                attn_input = hook_save("act_in", attn_input)
                q, k, v = block.attn.c_attn(attn_input).split(
                    block.attn.n_head * block.attn.d_head,
                    dim=2,
                )
                k = block.config.maybe_activation_sparsity(k, "attn_k")
                q = block.config.maybe_activation_sparsity(q, "attn_q")
                v = block.config.maybe_activation_sparsity(v, "attn_v")
                k = hook_save("k", k)
                q = hook_save("q", q)
                v = hook_save("v", v)
                q_heads = q.view(batch_size, 1, block.attn.n_head, block.attn.d_head).transpose(1, 2)
                k_final = k.view(batch_size, 1, block.attn.n_head, block.attn.d_head).transpose(1, 2)
                v_final = v.view(batch_size, 1, block.attn.n_head, block.attn.d_head).transpose(1, 2)
                prefix_k = cache.keys[layer_index].expand(batch_size, -1, -1, -1)
                prefix_v = cache.values[layer_index].expand(batch_size, -1, -1, -1)
                prefix_q = cache.queries[layer_index].expand(batch_size, -1, -1, -1)
                attention_output = _final_query_attention(
                    block.attn,
                    prefix_q,
                    prefix_k,
                    prefix_v,
                    q_heads,
                    k_final,
                    v_final,
                )
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size,
                    1,
                    block.attn.n_head * block.attn.d_head,
                )
                attention_output = hook_save("y", attention_output)
                attention_delta = block.attn.resid_dropout(block.attn.c_proj(attention_output))
                attention_delta = block.config.maybe_activation_sparsity(attention_delta, "attn_out")
                attention_delta = hook_save("resid_delta", attention_delta)
            hidden = hidden + attention_delta
            if block.config.residual_activation_type == "relu":
                hidden = torch.relu(hidden)
            hidden = block.config.maybe_activation_sparsity(hidden, "resid_post_attn")

            hidden = hook_save("resid_mid", hidden)
            with hook_namespace("mlp"):
                mlp_input = block.ln_2(hidden)
                if model.config.cat_pos_emb:
                    mlp_input = torch.cat((mlp_input, block.ln_p2(position_to_cat)), dim=-1)
                mlp_input = block.config.maybe_activation_sparsity(mlp_input, "mlp_in")
                mlp_input = hook_save("act_in", mlp_input)
                post_activation = block.mlp.c_fc(mlp_input)
                post_activation = block.mlp.act_fn(post_activation)
                post_activation = block.config.maybe_activation_sparsity(post_activation, "mlp_neuron")
                post_activation = hook_save("post_act", post_activation)
                mlp_delta = block.mlp.c_proj(post_activation)
                mlp_delta = block.mlp.dropout(mlp_delta)
                mlp_delta = block.config.maybe_activation_sparsity(mlp_delta, "mlp_out")
                mlp_delta = hook_save("resid_delta", mlp_delta)
            hidden = hidden + mlp_delta
            if block.config.residual_activation_type == "relu":
                hidden = torch.relu(hidden)
            hidden = block.config.maybe_activation_sparsity(hidden, "resid_post_mlp")

    hidden = hook_save("final_resid", hidden)
    if not include_margin:
        return None
    final_hidden = model.transformer.ln_f(hidden[:, 0, :])
    return _selected_output_margin(
        model,
        final_hidden,
        final_token,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
