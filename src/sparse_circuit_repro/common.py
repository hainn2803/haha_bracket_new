from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import repro_common


candidate_ids = repro_common.candidate_ids
load_json = repro_common.load_json
read_jsonl = repro_common.read_jsonl
release_root = repro_common.release_root
run_module = repro_common.run_module
sha256_file = repro_common.sha256_file
verify_manifest = repro_common.verify_manifest


def download_models() -> dict[str, Any]:
    root = release_root()
    metadata = load_json(root / "MODEL_ARTIFACTS.json")
    cache_dir = Path(os.environ.get("TIKTOKEN_CACHE_DIR", root / ".cache" / "data-gym-cache"))
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    results: list[dict[str, Any]] = []
    for model in metadata["models"]:
        artifacts = []
        for artifact in model["artifacts"]:
            cache_path = cache_dir / artifact["cache_key"]
            if cache_path.exists() and sha256_file(cache_path) == artifact["sha256"]:
                size = cache_path.stat().st_size
                digest = artifact["sha256"]
            else:
                size, digest = repro_common._download_stream(artifact["url"], cache_path)
            if size != int(artifact["bytes"]) or digest != artifact["sha256"]:
                cache_path.unlink(missing_ok=True)
                raise RuntimeError(f"download verification failed for {model['model_name']}:{artifact['name']}")
            artifacts.append(
                {
                    "name": artifact["name"],
                    "path": str(cache_path),
                    "bytes": size,
                    "sha256": digest,
                }
            )
        results.append({"model": model["model_name"], "artifacts": artifacts})
    payload = {"cache_dir": str(cache_dir), "models": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload
