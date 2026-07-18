from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CIRCUIT_REPO = "https://github.com/openai/circuit_sparsity.git"
CIRCUIT_COMMIT = "dbf1fe0d27b76c19e10d2a715f28c2e5da535e08"


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path | None = None) -> dict[str, Any]:
    root = release_root() if root is None else root
    manifest = root / "MANIFEST.sha256"
    failures: list[str] = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256_file(path)
        checked += 1
        if actual.lower() != expected.lower():
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise RuntimeError("release integrity check failed:\n" + "\n".join(failures))
    return {"checked_files": checked, "valid": True}


def candidate_ids(csv_path: Path) -> tuple[str, ...]:
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = []
    for row in rows:
        if row.get("node_id"):
            ids.append(str(row["node_id"]))
        elif row.get("site_id"):
            ids.append(str(row["site_id"]))
        elif row.get("hook_key") is not None and row.get("coordinate") is not None:
            ids.append(f"{row['hook_key']}:{row['coordinate']}")
        else:
            raise ValueError(f"cannot identify candidate site from CSV columns: {tuple(row)}")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate CSV contains duplicate site IDs")
    return tuple(ids)


def mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError(f"cannot average empty record set for {key}")
    return sum(float(bool(row[key])) for row in rows) / len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_module(module: str, args: Sequence[str], *, root: Path | None = None) -> None:
    root = release_root() if root is None else root
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TIKTOKEN_CACHE_DIR", str(root / ".cache" / "data-gym-cache"))
    command = [sys.executable, "-m", module, *map(str, args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def ensure_circuit_repo(root: Path | None = None) -> Path:
    root = release_root() if root is None else root
    target = root / ".external" / "circuit_sparsity"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", CIRCUIT_REPO, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "origin", CIRCUIT_COMMIT], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", CIRCUIT_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != CIRCUIT_COMMIT:
        raise RuntimeError(f"circuit_sparsity commit mismatch: {actual}")
    return target


def _download_stream(url: str, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    temporary.replace(destination)
    return size, digest.hexdigest()


def download_model(model_name: str, *, root: Path | None = None) -> dict[str, Any]:
    root = release_root() if root is None else root
    metadata = load_json(root / "MODEL_ARTIFACTS.json")
    if metadata["model_name"] != model_name:
        raise ValueError(f"release is configured for {metadata['model_name']}, not {model_name}")
    cache_dir = Path(os.environ.get("TIKTOKEN_CACHE_DIR", root / ".cache" / "data-gym-cache"))
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    results = []
    for artifact in metadata["artifacts"]:
        cache_path = cache_dir / artifact["cache_key"]
        if cache_path.exists() and sha256_file(cache_path) == artifact["sha256"]:
            size = cache_path.stat().st_size
            digest = artifact["sha256"]
        else:
            size, digest = _download_stream(artifact["url"], cache_path)
        if size != int(artifact["bytes"]) or digest != artifact["sha256"]:
            cache_path.unlink(missing_ok=True)
            raise RuntimeError(f"download verification failed for {artifact['name']}")
        results.append({"name": artifact["name"], "path": str(cache_path), "bytes": size, "sha256": digest})
    print(json.dumps({"model": model_name, "cache_dir": str(cache_dir), "artifacts": results}, indent=2))
    return {"cache_dir": str(cache_dir), "artifacts": results}


def common_environment(root: Path | None = None) -> dict[str, str]:
    root = release_root() if root is None else root
    env = os.environ.copy()
    env.setdefault("TIKTOKEN_CACHE_DIR", str(root / ".cache" / "data-gym-cache"))
    return env


def relation_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["relation"]), []).append(row)
    return groups
