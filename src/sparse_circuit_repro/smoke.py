from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from experiments.openai_sparse_plot.runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids

from .common import release_root


def _ids(enc: Any, prompt: str, cuda: bool) -> torch.Tensor:
    value = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
    return value.cuda() if cuda else value


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(*, cuda: bool) -> dict[str, Any]:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    enc = make_tinypython_encoding(circuit)

    quote_model, quote_info = load_sparse_gpt_model(
        model_name="csp_yolo1",
        circuit_home=circuit,
        cuda=cuda,
        flash=False,
        grad_checkpointing=False,
    )
    quote_ids = quote_token_ids(enc)
    quote_rows = []
    with torch.no_grad():
        for prompt, expected in (("x = \"abc", "double"), ("x = 'abc", "single")):
            ids = _ids(enc, prompt, cuda)
            logits, _, _ = quote_model(ids)
            last = logits[0, -1]
            predicted = "double" if float(last[quote_ids["double"]]) > float(last[quote_ids["single"]]) else "single"
            quote_rows.append({"prompt": prompt, "expected": expected, "predicted": predicted, "correct": predicted == expected})
    _release_model(quote_model)

    bracket_model, bracket_info = load_sparse_gpt_model(
        model_name="csp_yolo2",
        circuit_home=circuit,
        cuda=cuda,
        flash=True,
        grad_checkpointing=False,
    )
    one_id = int(enc.encode("]\n")[0])
    two_id = int(enc.encode("]]\n")[0])
    bracket_rows = []
    with torch.no_grad():
        for prompt, expected in (("values = [1, 9, 3, 11", "one"), ("values = [[1, 9, 3, 11", "two")):
            ids = _ids(enc, prompt, cuda)
            logits, _, _ = bracket_model(ids)
            last = logits[0, -1]
            predicted = "two" if float(last[two_id]) > float(last[one_id]) else "one"
            bracket_rows.append({"prompt": prompt, "expected": expected, "predicted": predicted, "correct": predicted == expected})
    _release_model(bracket_model)

    payload = {
        "quote": {"model": quote_info["model_name"], "records": quote_rows, "all_correct": all(row["correct"] for row in quote_rows)},
        "bracket": {"model": bracket_info["model_name"], "records": bracket_rows, "all_correct": all(row["correct"] for row in bracket_rows)},
    }
    if not payload["quote"]["all_correct"] or not payload["bracket"]["all_correct"]:
        raise RuntimeError("clean-model smoke test failed")
    out = root / "outputs" / "model_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload
