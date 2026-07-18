from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RELATIONS: tuple[str, ...] = (
    "same_D",
    "different_D_same_R",
    "different_D_different_R",
    "same_surface_different_active_context",
    "wrong_numeric_content",
    "wrong_tail_length",
)


DEFAULT_NUMERIC_CONTENTS: tuple[str, ...] = (
    "5, 3, 11, 3, 12",
    "9, 3",
    "1, 9, 3, 11",
    "3, 12",
    "3, 4",
    "2",
    "7, 2, 2, 10, 9, 2, 3",
    "11, 7",
    "5, 12, 2, 9, 14",
    "4, 7",
    "2, 8, 0, 3",
    "0, 0, 1, 5, 15",
    "13, 15",
    "10, 9",
    "1, 7, 6",
    "3, 7, 1, 1",
)


CONTEXT_FAMILIES: tuple[str, ...] = (
    "no_distractor",
    "closed_pair_before",
    "outside_active",
    "surface_balanced",
)


@dataclass(frozen=True)
class MultiDepthBracketExample:
    example_id: str
    prompt: str
    token_ids: tuple[int, ...]
    tail: str
    depth: int
    close_count: int
    split: str
    pair_id: str
    context_family: str
    numeric_content: str
    surface_open_count: int
    surface_close_count: int

    def sign(self) -> int:
        return 1 if self.close_count == 2 else -1


@dataclass(frozen=True)
class MultiDepthResamplingSpec:
    relation: str
    base_id: str
    source_id: str
    wrong_variable: str | None = None


def parse_depths(text: str) -> tuple[int, ...]:
    depths = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not depths:
        raise ValueError("at least one depth is required")
    if min(depths) < 1:
        raise ValueError("multi-depth bracket experiment expects positive active depths")
    return depths


def close_count_from_depth(depth: int) -> int:
    return 1 if int(depth) <= 1 else 2


def build_active_tail(depth: int, numeric_content: str) -> str:
    if depth < 1:
        raise ValueError("active depth must be positive")
    return "values =" + ("[" * int(depth)) + numeric_content


def depth_from_tail(tail: str) -> int:
    if "values =" not in tail:
        raise ValueError("tail must contain `values =`")
    active = tail[tail.rfind("values =") :]
    return int(active.count("[") - active.count("]"))


def numbers_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit() or ch in {",", " "})


def _encode_prompt(enc: Any | None, prompt: str) -> tuple[int, ...]:
    if enc is None:
        return tuple(ord(ch) for ch in prompt)
    return tuple(int(tok) for tok in enc.encode(prompt))


def _context_prefix(context_family: str, *, depth: int, max_depth: int, index: int) -> str:
    if context_family == "no_distractor":
        return ""
    if context_family == "closed_pair_before":
        return f"scratch_{index} = [[0], [1]]\n"
    if context_family == "outside_active":
        return f"outer_{index} = [9]\nmeta_{index} = [[2]]\n"
    if context_family == "surface_balanced":
        extra = max(0, int(max_depth) - int(depth))
        return "".join(f"pad_{index}_{j} = [{j}]\n" for j in range(extra))
    raise ValueError(f"unknown context family: {context_family}")


def generate_multidepth_examples(
    enc: Any | None,
    *,
    depths: Sequence[int],
    examples_per_depth: int,
    numeric_contents: Sequence[str] = DEFAULT_NUMERIC_CONTENTS,
    context_families: Sequence[str] = CONTEXT_FAMILIES,
) -> tuple[MultiDepthBracketExample, ...]:
    if examples_per_depth <= 0:
        raise ValueError("examples_per_depth must be positive")
    depths = tuple(int(depth) for depth in depths)
    if not depths or min(depths) < 1:
        raise ValueError("depths must be nonempty positive integers")
    if not numeric_contents:
        raise ValueError("numeric_contents must be nonempty")
    if not context_families:
        raise ValueError("context_families must be nonempty")

    max_depth = max(depths)
    examples: list[MultiDepthBracketExample] = []
    for depth in depths:
        for idx in range(int(examples_per_depth)):
            content = str(numeric_contents[idx % len(numeric_contents)])
            context_family = str(context_families[idx % len(context_families)])
            prefix = _context_prefix(context_family, depth=depth, max_depth=max_depth, index=idx)
            tail = build_active_tail(depth, content)
            prompt = prefix + tail
            token_ids = _encode_prompt(enc, prompt)
            split = "calibration" if idx % 2 == 0 else "heldout"
            pair_id = f"content-{idx % len(numeric_contents):02d}-{context_family}"
            examples.append(
                MultiDepthBracketExample(
                    example_id=f"d{depth}-{idx:02d}-{context_family}",
                    prompt=prompt,
                    token_ids=token_ids,
                    tail=tail,
                    depth=int(depth),
                    close_count=close_count_from_depth(depth),
                    split=split,
                    pair_id=pair_id,
                    context_family=context_family,
                    numeric_content=content,
                    surface_open_count=prompt.count("["),
                    surface_close_count=prompt.count("]"),
                )
            )
    return tuple(examples)


def relation_for_pair(
    base: MultiDepthBracketExample,
    source: MultiDepthBracketExample,
    relation: str,
) -> MultiDepthResamplingSpec | None:
    if base.example_id == source.example_id:
        return None
    same_d = base.depth == source.depth
    same_r = base.close_count == source.close_count
    if relation == "same_D":
        if same_d and base.pair_id != source.pair_id:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "different_D_same_R":
        if (not same_d) and same_r:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "different_D_different_R":
        if not same_r:
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id)
    elif relation == "same_surface_different_active_context":
        if (
            not same_d
            and base.surface_open_count == source.surface_open_count
            and base.context_family != source.context_family
        ):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "active_context")
    elif relation == "wrong_numeric_content":
        if same_d and base.context_family == source.context_family and numbers_only(base.tail) != numbers_only(source.tail):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "numeric_content")
    elif relation == "wrong_tail_length":
        if same_d and base.context_family == source.context_family and len(base.tail) != len(source.tail):
            return MultiDepthResamplingSpec(relation, base.example_id, source.example_id, "tail_length")
    else:
        raise ValueError(f"unknown relation: {relation}")
    return None


def build_relation_specs(
    examples: Sequence[MultiDepthBracketExample],
    *,
    split: str,
    max_records_per_relation: int,
    relations: Sequence[str] = DEFAULT_RELATIONS,
) -> tuple[MultiDepthResamplingSpec, ...]:
    split_examples = sorted((ex for ex in examples if ex.split == split), key=lambda ex: ex.example_id)
    by_id = {ex.example_id: ex for ex in split_examples}
    out: list[MultiDepthResamplingSpec] = []
    for relation in relations:
        specs: list[MultiDepthResamplingSpec] = []
        for base in split_examples:
            for source in split_examples:
                spec = relation_for_pair(base, source, relation)
                if spec is not None:
                    specs.append(spec)
        out.extend(_balanced_prefix(specs, by_id, int(max_records_per_relation)))
    return tuple(out)


def _balanced_prefix(
    specs: Sequence[MultiDepthResamplingSpec],
    examples_by_id: Mapping[str, MultiDepthBracketExample],
    limit: int,
) -> list[MultiDepthResamplingSpec]:
    if limit <= 0:
        return []
    if len(specs) <= limit:
        return list(specs)
    by_base_depth: dict[int, list[MultiDepthResamplingSpec]] = defaultdict(list)
    for spec in specs:
        by_base_depth[examples_by_id[spec.base_id].depth].append(spec)
    out: list[MultiDepthResamplingSpec] = []
    depths = sorted(by_base_depth)
    while len(out) < limit and any(by_base_depth.values()):
        for depth in depths:
            rows = by_base_depth[depth]
            if rows and len(out) < limit:
                out.append(rows.pop(0))
    return out


def relation_counts(specs: Iterable[MultiDepthResamplingSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec.relation] += 1
    return dict(sorted(counts.items()))
