from __future__ import annotations

import io
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .artifacts import add_circuit_sparsity_to_path


def filter_gpt_config(raw_config: dict[str, Any], *, flash: bool, grad_checkpointing: bool) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Filter public OpenAI model configs to the checked-out GPTConfig surface."""

    from circuit_sparsity.inference.gpt import GPTConfig

    config = dict(raw_config)
    if "n_mlp" in config:
        config["d_mlp"] = config.pop("n_mlp")
    config["flash"] = flash
    config["grad_checkpointing"] = grad_checkpointing
    if "sink" not in config:
        config["sink"] = False
    if "use_tied_aux_matrix" in config:
        value = config.pop("use_tied_aux_matrix")
        if value:
            raise ValueError("use_tied_aux_matrix=True is not supported by this loader")

    allowed = {field.name for field in fields(GPTConfig)}
    dropped = tuple(sorted(k for k in config if k not in allowed))
    filtered = {k: v for k, v in config.items() if k in allowed}
    return filtered, dropped


def load_sparse_gpt_model(
    *,
    model_name: str = "csp_yolo1",
    circuit_home: str | Path | None = None,
    cuda: bool = False,
    flash: bool = False,
    grad_checkpointing: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a public OpenAI sparse GPT model with config-key compatibility filtering."""

    add_circuit_sparsity_to_path(circuit_home)

    import blobfile as bf
    import torch
    from tiktoken.load import read_file_cached

    from circuit_sparsity.inference.gpt import GPT, GPTConfig
    from circuit_sparsity.registries import MODEL_BASE_DIR

    model_path = f"{MODEL_BASE_DIR}/models/{model_name}"
    raw_config = json.loads(read_file_cached(f"{model_path}/beeg_config.json").decode())
    config_dict, dropped_keys = filter_gpt_config(
        raw_config,
        flash=flash,
        grad_checkpointing=grad_checkpointing,
    )
    if config_dict.get("sink") and not flash:
        raise ValueError(
            "This model uses attention sinks; load it with flash=True so SDPAWithSink is active."
        )
    config = GPTConfig(**config_dict)
    model = GPT(config)
    ckpt_path = bf.join(model_path, "final_model.pt")
    map_location = "cuda" if cuda else "cpu"
    state_dict = torch.load(
        io.BytesIO(read_file_cached(ckpt_path)),
        weights_only=True,
        map_location=map_location,
    )
    if "final_logits_bias" not in state_dict:
        state_dict["final_logits_bias"] = torch.zeros(config.vocab_size)
    load_result = model.load_state_dict(state_dict, strict=False)
    if cuda:
        model.cuda()
    model.eval()
    info = {
        "model_name": model_name,
        "model_path": model_path,
        "raw_config": raw_config,
        "filtered_config": config_dict,
        "dropped_config_keys": dropped_keys,
        "missing_keys": tuple(load_result.missing_keys),
        "unexpected_keys": tuple(load_result.unexpected_keys),
    }
    return model, info


def make_tinypython_encoding(circuit_home: str | Path | None = None) -> Any:
    add_circuit_sparsity_to_path(circuit_home)
    from tiktoken import Encoding

    from circuit_sparsity.tiktoken_ext import tinypython

    return Encoding(**tinypython.tinypython_2k())


def quote_token_ids(enc: Any) -> dict[str, int]:
    return {
        "single": int(enc.encode("'")[0]),
        "double": int(enc.encode('"')[0]),
        "single_newline": int(enc.encode("'\n")[0]),
        "double_newline": int(enc.encode('"\n')[0]),
    }
