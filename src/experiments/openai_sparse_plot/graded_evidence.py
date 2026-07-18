from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .ablate_rediscover import RediscoveryExample, RediscoveryPair, _balanced_prefix
from .bracket_group_depth_plot import build_unique_discovery_bank


@dataclass(frozen=True)
class AffineDecoder:
    slope: float
    intercept: float
    fit_r2: float
    fit_pearson: float

    def predict(self, value: float) -> float:
        return float(self.slope) * float(value) + float(self.intercept)

    def to_dict(self) -> dict[str, float]:
        return {
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "fit_r2": float(self.fit_r2),
            "fit_pearson": float(self.fit_pearson),
        }


def fit_affine_decoder(values: Sequence[float], targets: Sequence[float]) -> AffineDecoder:
    if len(values) != len(targets) or len(values) < 2:
        raise ValueError("affine decoder data must be aligned and nontrivial")
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    design = np.stack((x, np.ones_like(x)), axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else float("nan")
    pearson = float(np.corrcoef(prediction, y)[0, 1]) if np.std(prediction) > 0 and np.std(y) > 0 else 0.0
    return AffineDecoder(float(slope), float(intercept), r2, pearson)


def decoder_metrics(decoder: AffineDecoder, values: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    prediction = np.asarray([decoder.predict(value) for value in values], dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return {
        "r2": 1.0 - residual / total if total > 0.0 else float("nan"),
        "pearson": float(np.corrcoef(prediction, target)[0, 1])
        if np.std(prediction) > 0 and np.std(target) > 0
        else 0.0,
        "mae": float(np.mean(np.abs(prediction - target))),
    }


def neutral_prefix(content_key: str, q: int) -> str:
    return "".join(f"neutral_{content_key}_{index} = 0\n" for index in range(int(q)))


def build_graded_evidence_bank(
    enc: Any,
    *,
    fit_contents: int,
    cal_contents: int,
    test_contents: int,
    content_offset: int,
    q_grid: Sequence[int],
) -> tuple[RediscoveryExample, ...]:
    base_rows = build_unique_discovery_bank(
        enc,
        contents=int(fit_contents) + int(cal_contents) + int(test_contents),
        fit_contents=int(fit_contents),
        cal_contents=int(cal_contents),
        test_contents=int(test_contents),
        content_offset=int(content_offset),
        depths=(1, 2, 3, 4),
    )
    output: list[RediscoveryExample] = []
    for row in base_rows:
        content_key = str(row.numeric_content).replace(",", "_").replace(" ", "")
        for q in tuple(int(value) for value in q_grid):
            prompt = neutral_prefix(content_key, q) + row.prompt
            token_ids = tuple(int(value) for value in enc.encode(prompt))
            length = len(token_ids)
            total_open = prompt.count("[")
            output.append(
                RediscoveryExample(
                    example_id=f"{row.example_id}-q{q}",
                    prompt=prompt,
                    token_ids=token_ids,
                    split=row.split,
                    content_id=str(row.numeric_content),
                    variable_value=1 if int(row.close_count) == 2 else -1,
                    patch_position=length - 1,
                    metadata={
                        "task": "bracket_graded_evidence",
                        "depth": int(row.depth),
                        "close_count": int(row.close_count),
                        "context_family": row.context_family,
                        "numeric_content": row.numeric_content,
                        "q": q,
                        "base_key": row.example_id,
                        "context_length": length,
                        "surface_open_count": total_open,
                        "surface_density": float(total_open) / float(length),
                        "active_density": float(row.depth) / float(length),
                        "active_depth": float(row.depth),
                    },
                )
            )
    return tuple(output)


def e_value(example: RediscoveryExample, definition: str) -> float:
    if definition not in {"surface_density", "active_density", "active_depth"}:
        raise ValueError(f"unknown graded evidence definition: {definition}")
    return float(example.metadata[definition])


def build_graded_pairs(
    examples: Sequence[RediscoveryExample],
    *,
    split: str,
    records_per_relation: int,
    e_definition: str,
) -> tuple[RediscoveryPair, ...]:
    rows = sorted((row for row in examples if row.split == split), key=lambda row: row.example_id)
    by_id = {row.example_id: row for row in rows}
    output: list[RediscoveryPair] = []
    for relation in (
        "different_R",
        "same_R_different_E",
        "same_D_padding",
        "wrong_numeric_content",
    ):
        candidates: list[RediscoveryPair] = []
        for base in rows:
            for source in rows:
                if base.example_id == source.example_id:
                    continue
                same_r = base.variable_value == source.variable_value
                different_e = abs(e_value(base, e_definition) - e_value(source, e_definition)) > 1e-9
                same_base = str(base.metadata["base_key"]) == str(source.metadata["base_key"])
                same_depth = int(base.metadata["depth"]) == int(source.metadata["depth"])
                same_q = int(base.metadata["q"]) == int(source.metadata["q"])
                same_context = str(base.metadata["context_family"]) == str(source.metadata["context_family"])
                different_content = str(base.metadata["numeric_content"]) != str(source.metadata["numeric_content"])
                keep = (
                    (relation == "different_R" and not same_r)
                    or (relation == "same_R_different_E" and same_r and different_e and not same_base)
                    or (
                        relation == "same_D_padding"
                        and same_depth
                        and same_base
                        and int(base.metadata["q"]) != int(source.metadata["q"])
                    )
                    or (
                        relation == "wrong_numeric_content"
                        and same_depth
                        and same_q
                        and same_context
                        and different_content
                    )
                )
                if keep:
                    candidates.append(RediscoveryPair(relation, base.example_id, source.example_id))
        output.extend(_balanced_prefix(candidates, by_id, int(records_per_relation)))
    return tuple(output)


def abstract_e_signature(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    *,
    definition: str,
) -> tuple[float, ...]:
    return tuple(e_value(examples[pair.source_id], definition) - e_value(examples[pair.base_id], definition) for pair in pairs)


def moves_continuous(base: float, source: float, patched: float) -> bool:
    if abs(float(source) - float(base)) <= 1e-10:
        return abs(float(patched) - float(base)) <= 1e-6
    return abs(float(source) - float(patched)) < abs(float(source) - float(base))


def graded_validation_summary(
    pairs: Sequence[RediscoveryPair],
    examples: Mapping[str, RediscoveryExample],
    abstract_e_by_id: Mapping[str, float],
    clean_e_by_id: Mapping[str, float],
    patched_e: Sequence[float],
    patched_rmid_state: Sequence[int],
    patched_output_state: Sequence[int],
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, bool]]] = {}
    cross: list[dict[str, bool]] = []
    same_side: list[dict[str, bool]] = []
    for index, pair in enumerate(pairs):
        base = examples[pair.base_id]
        source = examples[pair.source_id]
        expected_r = source.variable_value if source.variable_value != base.variable_value else base.variable_value
        expected_e = float(abstract_e_by_id[pair.source_id])
        base_abstract_e = float(abstract_e_by_id[pair.base_id])
        if all(abs(value - round(value)) <= 1e-8 for value in abstract_e_by_id.values()):
            e_correct = int(round(float(patched_e[index]))) == int(round(expected_e))
        elif abs(expected_e - base_abstract_e) <= 1e-10:
            e_correct = abs(float(patched_e[index]) - expected_e) <= 0.10
        else:
            e_correct = abs(float(patched_e[index]) - expected_e) < abs(
                float(clean_e_by_id[pair.base_id]) - expected_e
            )
        record = {
            "E_moves": moves_continuous(
                clean_e_by_id[pair.base_id], clean_e_by_id[pair.source_id], float(patched_e[index])
            ),
            "E_correct": e_correct,
            "Rmid_correct": int(patched_rmid_state[index]) == int(expected_r),
            "output_correct": int(patched_output_state[index]) == int(expected_r),
        }
        rows.setdefault(pair.relation, []).append(record)
        (cross if source.variable_value != base.variable_value else same_side).append(record)
    relation_metrics = {
        relation: {
            key: float(np.mean([row[key] for row in relation_rows]))
            for key in ("E_moves", "E_correct", "Rmid_correct", "output_correct")
        }
        for relation, relation_rows in sorted(rows.items())
    }
    blocks = {
        "E_correct": float(np.mean([row["E_correct"] for relation_rows in rows.values() for row in relation_rows])),
        "cross_Rmid": float(np.mean([row["Rmid_correct"] for row in cross])),
        "cross_output": float(np.mean([row["output_correct"] for row in cross])),
        "same_side_Rmid": float(np.mean([row["Rmid_correct"] for row in same_side])),
        "same_side_output": float(np.mean([row["output_correct"] for row in same_side])),
    }
    return {
        "relations": relation_metrics,
        "balanced_blocks": blocks,
        "score": float(np.mean(list(blocks.values()))),
        "passes": bool(min(blocks.values()) >= 0.90),
    }


def decoded_values(values: np.ndarray, decoder: AffineDecoder, probe_index: int) -> np.ndarray:
    return decoder.slope * values[..., int(probe_index)] + decoder.intercept


def safe_correlation(values: Sequence[float], targets: Sequence[float]) -> float:
    if len(values) < 2 or np.std(values) <= 0 or np.std(targets) <= 0:
        return 0.0
    return float(np.corrcoef(values, targets)[0, 1])
