from __future__ import annotations

import csv
import importlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from .schema import SparseCircuitEdge, SparseCircuitGraph, SparseCircuitNode


DEFAULT_MODEL = "csp_yolo1"
DEFAULT_TASK = "single_double_quote"
DEFAULT_SWEEPS = ("prune_v2", "prune_v5_logitscaling", "prune_v4", "prune_v3")
DEFAULT_K_CANDIDATES = (64, 128, 256, 512, 1024, "k_optim", 12, 16, 20, 24, 32)


class CircuitSparsityUnavailable(RuntimeError):
    pass


def add_circuit_sparsity_to_path(circuit_home: str | Path | None = None) -> Path | None:
    home = circuit_home or os.environ.get("CIRCUIT_SPARSITY_HOME")
    if not home:
        return None
    path = Path(home).expanduser().resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def import_circuit_sparsity(circuit_home: str | Path | None = None) -> Any:
    add_circuit_sparsity_to_path(circuit_home)
    try:
        return importlib.import_module("circuit_sparsity")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise CircuitSparsityUnavailable(
            "Could not import circuit_sparsity. Install OpenAI's repo or set CIRCUIT_SPARSITY_HOME."
        ) from exc


def circuit_sparsity_status(circuit_home: str | Path | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "import_ok": False,
        "circuit_home": str(add_circuit_sparsity_to_path(circuit_home) or ""),
        "model_base_dir": "",
        "cache_dir": "",
        "error": "",
    }
    try:
        import_circuit_sparsity(circuit_home)
        registries = importlib.import_module("circuit_sparsity.registries")
        status.update(
            {
                "import_ok": True,
                "model_base_dir": str(getattr(registries, "MODEL_BASE_DIR", "")),
                "cache_dir": str(getattr(registries, "CACHE_DIR", "")),
            }
        )
    except Exception as exc:
        status["error"] = str(exc)
    return status


def candidate_viz_paths(
    *,
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    sweeps: tuple[str, ...] = DEFAULT_SWEEPS,
    ks: tuple[int | str, ...] = DEFAULT_K_CANDIDATES,
    base_dir: str | None = None,
) -> tuple[str, ...]:
    if base_dir is None:
        base_dir = "https://openaipublic.blob.core.windows.net/circuit-sparsity"
    paths: list[str] = []
    for sweep in sweeps:
        for k in ks:
            for suffix in ("viz_data.pt", "viz_data.pkl"):
                paths.append(f"{base_dir}/viz/{model}/{task}/{sweep}/{k}/{suffix}")
    return tuple(dict.fromkeys(paths))


def _read_bytes(path_or_url: str | Path) -> bytes:
    text = str(path_or_url)
    if text.startswith("http://") or text.startswith("https://"):
        try:
            from tiktoken.load import read_file_cached
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise CircuitSparsityUnavailable("tiktoken is required to read remote OpenAI blob artifacts") from exc
        return read_file_cached(text)
    return Path(text).expanduser().read_bytes()


def load_viz_data(path_or_url: str | Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - project requires torch
        raise RuntimeError("torch is required to load viz_data artifacts") from exc
    payload = _read_bytes(path_or_url)
    try:
        return torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def summarize_value(value: Any, *, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k): summarize_value(v, depth=depth + 1, max_depth=max_depth) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": [summarize_value(v, depth=depth + 1, max_depth=max_depth) for v in list(value)[:5]],
        }
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": [int(x) for x in shape]}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def summarize_viz_data(viz_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_level_keys": sorted(str(k) for k in viz_data.keys()),
        "summary": summarize_value(viz_data, max_depth=4),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _as_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return [value]


def _node_id(location: str, index: Any) -> str:
    return f"{location}:{index}"


def _parse_location(location: str) -> tuple[int | None, str, str]:
    parts = str(location).split(".")
    if len(parts) >= 3 and parts[0].isdigit():
        return int(parts[0]), parts[1], ".".join(parts[2:])
    return None, "unknown", str(location)


def _importance_at(ch_losses: Any, row_index: int) -> float | None:
    values = _as_python_list(ch_losses)
    if 0 <= row_index < len(values):
        return _float_or_none(values[row_index])
    return _float_or_none(ch_losses)


def _add_retained_nodes(
    nodes: dict[str, SparseCircuitNode],
    retained_nodes: dict[str, Any],
    ch_losses: dict[str, Any],
) -> None:
    for location, raw_indices in retained_nodes.items():
        if location == "prune_config":
            continue
        indices = _as_python_list(raw_indices)
        layer, module, node_kind = _parse_location(str(location))
        for row_index, channel in enumerate(indices):
            if channel == "bias":
                index = None
                kind = "bias"
            else:
                index = int(channel)
                kind = node_kind
            node_id = _node_id(str(location), channel)
            nodes[node_id] = SparseCircuitNode(
                node_id=node_id,
                layer=layer,
                module=module,
                node_kind=kind,
                index=index,
                source_key=str(location),
                importance=_importance_at(ch_losses.get(location), row_index),
                metadata={"row_index": row_index, "raw_channel": str(channel)},
            )


def _add_pair_data_edges(
    nodes: dict[str, SparseCircuitNode],
    edges: list[SparseCircuitEdge],
    pair_data: Any,
) -> int:
    parsed_entries = 0
    entries = _as_python_list(pair_data)
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            continue
        weights, src_names, dst_names, loc_pair = entry[:4]
        if not isinstance(loc_pair, (list, tuple)) or len(loc_pair) != 2:
            continue
        src_location, dst_location = str(loc_pair[0]), str(loc_pair[1])
        src_channels = _as_python_list(src_names)
        dst_channels = _as_python_list(dst_names)
        parsed_entries += 1

        if hasattr(weights, "detach"):
            weights = weights.detach().cpu()
        if hasattr(weights, "nonzero") and hasattr(weights, "shape"):
            nonzero = weights.nonzero()
            if hasattr(nonzero, "tolist"):
                nonzero = nonzero.tolist()
            for ij in nonzero:
                if len(ij) < 2:
                    continue
                i, j = int(ij[0]), int(ij[1])
                if i >= len(src_channels) or j >= len(dst_channels):
                    continue
                src_channel = src_channels[i]
                dst_channel = dst_channels[j]
                src_id = _node_id(src_location, src_channel)
                dst_id = _node_id(dst_location, dst_channel)
                for node_id, location, channel in (
                    (src_id, src_location, src_channel),
                    (dst_id, dst_location, dst_channel),
                ):
                    if node_id not in nodes:
                        layer, module, kind = _parse_location(location)
                        nodes[node_id] = SparseCircuitNode(
                            node_id=node_id,
                            layer=layer,
                            module=module,
                            node_kind="bias" if channel == "bias" else kind,
                            index=None if channel == "bias" else int(channel),
                            source_key=location,
                            metadata={"created_from": "pair_data"},
                        )
                edges.append(
                    SparseCircuitEdge(
                        src=src_id,
                        dst=dst_id,
                        weight=_float_or_none(weights[i, j]),
                        edge_kind="pair_data",
                        metadata={
                            "entry_index": entry_index,
                            "src_location": src_location,
                            "dst_location": dst_location,
                            "src_row": i,
                            "dst_row": j,
                        },
                    )
                )
        elif isinstance(weights, (list, tuple)):
            for i, row in enumerate(weights):
                for j, value in enumerate(_as_python_list(row)):
                    if not value or i >= len(src_channels) or j >= len(dst_channels):
                        continue
                    src_id = _node_id(src_location, src_channels[i])
                    dst_id = _node_id(dst_location, dst_channels[j])
                    edges.append(
                        SparseCircuitEdge(
                            src=src_id,
                            dst=dst_id,
                            weight=_float_or_none(value),
                            edge_kind="pair_data",
                            metadata={"entry_index": entry_index},
                        )
                    )
    return parsed_entries


def graph_from_viz_data(
    viz_data: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    sweep: str | None = None,
    k: int | None = None,
    source_artifact: str | None = None,
) -> SparseCircuitGraph:
    """Best-effort graph export from OpenAI visualizer payloads.

    The OpenAI payload schema is not treated as a stable public API. This
    extractor records whatever node/edge-like structures are present and writes
    diagnostics into graph.metadata when exact circuit edges cannot be inferred.
    """

    importances = viz_data.get("importances", viz_data)
    nodes: dict[str, SparseCircuitNode] = {}
    edges: list[SparseCircuitEdge] = []

    ch_losses = importances.get("ch_interv_losses", {})
    retained_nodes = viz_data.get("circuit_data", {})
    if isinstance(retained_nodes, dict):
        _add_retained_nodes(nodes, retained_nodes, ch_losses if isinstance(ch_losses, dict) else {})

    parsed_pair_entries = _add_pair_data_edges(nodes, edges, importances.get("pair_data", ()))

    if isinstance(ch_losses, dict):
        for key, value in ch_losses.items():
            if str(key) in retained_nodes:
                continue
            node_id = str(key)
            importance = None
            shape = getattr(value, "shape", None)
            if shape is None:
                importance = _float_or_none(value)
            nodes[node_id] = SparseCircuitNode(
                node_id=node_id,
                layer=None,
                module="unknown",
                node_kind="channel_or_node",
                source_key=str(key),
                importance=importance,
                metadata={"raw_summary": summarize_value(value, max_depth=2)},
            )

    if parsed_pair_entries == 0:
        pair_connections = importances.get("pair_data_connections", ())
        if isinstance(pair_connections, dict):
            iterable = pair_connections.items()
        else:
            iterable = enumerate(pair_connections) if isinstance(pair_connections, (list, tuple)) else ()
        for key, item in iterable:
            if isinstance(item, dict):
                src = item.get("src") or item.get("source") or item.get("from")
                dst = item.get("dst") or item.get("target") or item.get("to")
                weight = item.get("weight") or item.get("delta") or item.get("importance")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                src, dst = item[0], item[1]
                weight = item[2] if len(item) >= 3 else None
            else:
                continue
            if src is None or dst is None:
                continue
            src_id = str(src)
            dst_id = str(dst)
            for node_id in (src_id, dst_id):
                nodes.setdefault(
                    node_id,
                    SparseCircuitNode(
                        node_id=node_id,
                        layer=None,
                        module="unknown",
                        node_kind="connection_endpoint",
                        source_key=str(node_id),
                    ),
                )
            edges.append(
                SparseCircuitEdge(
                    src=src_id,
                    dst=dst_id,
                    weight=_float_or_none(weight),
                    edge_kind="pair_data_connections",
                    metadata={"source_connection_key": str(key)},
                )
            )

    graph = SparseCircuitGraph(
        model=model,
        task=task,
        sweep=sweep,
        k=k,
        source_artifact=source_artifact,
        nodes=tuple(nodes.values()),
        edges=tuple(edges),
        metadata={
            "viz_data_summary": summarize_viz_data(viz_data),
            "retained_node_locations": sorted(str(k) for k in retained_nodes.keys())
            if isinstance(retained_nodes, dict)
            else [],
            "parsed_pair_data_entries": parsed_pair_entries,
            "extraction_warning": (
                "Best-effort extraction. Verify node/edge semantics against circuit_sparsity/viz.py "
                "and the Streamlit visualizer before using this graph for claims."
            ),
        },
    )
    graph.validate()
    return graph


def write_graph_tables(graph: SparseCircuitGraph, *, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.write_json(out / "string_closing_circuit.json")
    with (out / "string_closing_circuit_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_id",
                "layer",
                "module",
                "node_kind",
                "index",
                "token_position_role",
                "published_label",
                "source_key",
                "importance",
            ],
        )
        writer.writeheader()
        for node in graph.nodes:
            row = node.to_dict()
            row["index"] = json.dumps(row["index"])
            row.pop("metadata", None)
            writer.writerow(row)
    with (out / "string_closing_circuit_edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "dst", "weight", "edge_kind"])
        writer.writeheader()
        for edge in graph.edges:
            writer.writerow(
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "weight": edge.weight,
                    "edge_kind": edge.edge_kind,
                }
            )


def write_inventory_report(
    *,
    out_dir: str | Path,
    status: dict[str, Any],
    graph: SparseCircuitGraph | None = None,
    candidates: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "graph": None if graph is None else graph.to_dict(),
        "candidate_viz_paths": list(candidates),
        "notes": list(notes),
    }
    (out / "inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# OpenAI Sparse PLOT Inventory",
        "",
        f"- circuit_sparsity import: {'OK' if status.get('import_ok') else 'MISSING'}",
        f"- circuit_sparsity home: `{status.get('circuit_home', '')}`",
        f"- model base dir: `{status.get('model_base_dir', '')}`",
        f"- cache dir: `{status.get('cache_dir', '')}`",
    ]
    if status.get("error"):
        lines.append(f"- error: `{status['error']}`")
    if graph is None:
        lines.extend(
            [
                "",
                "## Circuit Export",
                "",
                "No viz artifact was loaded, so no circuit graph was exported.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Circuit Export",
                "",
                f"- model: `{graph.model}`",
                f"- task: `{graph.task}`",
                f"- sweep: `{graph.sweep}`",
                f"- k: `{graph.k}`",
                f"- nodes: `{len(graph.nodes)}`",
                f"- edges: `{len(graph.edges)}`",
                f"- source artifact: `{graph.source_artifact}`",
            ]
        )
    if candidates:
        lines.extend(["", "## Candidate Viz Paths", ""])
        lines.extend(f"- `{path}`" for path in candidates[:20])
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    (out / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
