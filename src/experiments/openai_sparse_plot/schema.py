from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence


QuoteType = Literal["single", "double"]


@dataclass(frozen=True)
class SparseCircuitNode:
    """A native node from a weight-sparse circuit artifact."""

    node_id: str
    layer: int | None
    module: str
    node_kind: str
    index: int | tuple[int, ...] | None = None
    token_position_role: str | None = None
    published_label: str | None = None
    source_key: str | None = None
    importance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.node_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparseCircuitNode":
        index = data.get("index")
        if isinstance(index, list):
            index = tuple(int(x) for x in index)
        return cls(
            node_id=str(data["node_id"]),
            layer=None if data.get("layer") is None else int(data["layer"]),
            module=str(data.get("module", "unknown")),
            node_kind=str(data.get("node_kind", "unknown")),
            index=index,
            token_position_role=data.get("token_position_role"),
            published_label=data.get("published_label"),
            source_key=data.get("source_key"),
            importance=None if data.get("importance") is None else float(data["importance"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SparseCircuitEdge:
    """A directed weighted edge between native sparse-circuit nodes."""

    src: str
    dst: str
    weight: float | None = None
    edge_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparseCircuitEdge":
        return cls(
            src=str(data["src"]),
            dst=str(data["dst"]),
            weight=None if data.get("weight") is None else float(data["weight"]),
            edge_kind=data.get("edge_kind"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SparseCircuitGraph:
    """Exported sparse circuit graph plus provenance."""

    model: str
    task: str
    nodes: tuple[SparseCircuitNode, ...]
    edges: tuple[SparseCircuitEdge, ...]
    sweep: str | None = None
    k: int | None = None
    source_artifact: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> set[str]:
        return {node.node_id for node in self.nodes}

    def validate(self) -> None:
        node_ids = self.node_ids()
        if len(node_ids) != len(self.nodes):
            raise ValueError("duplicate node_id in SparseCircuitGraph")
        missing = sorted(
            {edge.src for edge in self.edges if edge.src not in node_ids}
            | {edge.dst for edge in self.edges if edge.dst not in node_ids}
        )
        if missing:
            raise ValueError(f"edges reference missing nodes: {missing}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "sweep": self.sweep,
            "k": self.k,
            "source_artifact": self.source_artifact,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparseCircuitGraph":
        graph = cls(
            model=str(data["model"]),
            task=str(data["task"]),
            sweep=data.get("sweep"),
            k=None if data.get("k") is None else int(data["k"]),
            source_artifact=data.get("source_artifact"),
            metadata=dict(data.get("metadata", {})),
            nodes=tuple(SparseCircuitNode.from_dict(x) for x in data.get("nodes", [])),
            edges=tuple(SparseCircuitEdge.from_dict(x) for x in data.get("edges", [])),
        )
        graph.validate()
        return graph

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "SparseCircuitGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class StringClosingExample:
    """One string-closing prompt and its binary target."""

    example_id: str
    prompt: str
    opening_quote_type: QuoteType
    target_closing_type: QuoteType
    target_token: str
    alt_token: str
    content: str
    template_id: str
    split: str
    pair_id: str
    prompt_tokens: tuple[str, ...] = ()
    token_ids: tuple[int, ...] = ()
    target_token_id: int | None = None
    alt_token_id: int | None = None
    opening_quote_position: int | None = None
    final_position: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def sign(self) -> int:
        return 1 if self.opening_quote_type == "double" else -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SparseCircuitSite:
    """An intervention candidate: one native node or a small node group."""

    site_id: str
    node_ids: tuple[str, ...]
    site_kind: str = "node"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.site_id

    @classmethod
    def from_node(cls, node: SparseCircuitNode) -> "SparseCircuitSite":
        return cls(
            site_id=node.node_id,
            node_ids=(node.node_id,),
            site_kind="node",
            label=node.published_label,
        )


@dataclass(frozen=True)
class EffectSignatureTable:
    """Abstract and neural signatures aligned by a PLOT cost matrix."""

    abstract_variable_ids: tuple[str, ...]
    neural_site_ids: tuple[str, ...]
    abstract_signatures: tuple[tuple[float, ...], ...]
    neural_signatures: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.abstract_variable_ids) != len(self.abstract_signatures):
            raise ValueError("abstract ids/signatures length mismatch")
        if len(self.neural_site_ids) != len(self.neural_signatures):
            raise ValueError("neural ids/signatures length mismatch")
        expected = len(self.feature_names)
        for row in [*self.abstract_signatures, *self.neural_signatures]:
            if len(row) != expected:
                raise ValueError("all signatures must match feature_names length")

    @classmethod
    def from_sequences(
        cls,
        *,
        abstract_variable_ids: Sequence[str],
        neural_site_ids: Sequence[str],
        abstract_signatures: Sequence[Sequence[float]],
        neural_signatures: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> "EffectSignatureTable":
        table = cls(
            abstract_variable_ids=tuple(str(x) for x in abstract_variable_ids),
            neural_site_ids=tuple(str(x) for x in neural_site_ids),
            abstract_signatures=tuple(tuple(float(v) for v in row) for row in abstract_signatures),
            neural_signatures=tuple(tuple(float(v) for v in row) for row in neural_signatures),
            feature_names=tuple(str(x) for x in feature_names),
            metadata=dict(metadata or {}),
        )
        table.validate()
        return table
