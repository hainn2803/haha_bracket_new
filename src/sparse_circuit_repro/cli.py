from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .audit import audit
from .common import download_models, load_json, release_root, run_module


EXPERIMENTS = (
    "necessity-quote",
    "necessity-bracket",
    "rediscover-quote",
    "rediscover-bracket",
    "progressive-rmid",
    "graded-depth",
)


def _arguments(payload: Mapping[str, Any], *, cuda: bool) -> list[str]:
    result: list[str] = []
    for key, value in payload["arguments"].items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend((flag, str(value)))
    if cuda:
        result.append("--cuda")
    return result


def _run_experiment(name: str, *, cuda: bool) -> None:
    root = release_root()
    config = load_json(root / "configs" / f"{name}.json")
    run_module(config["module"], _arguments(config, cuda=cuda), root=root)


def _report(destination: Path | None) -> None:
    root = release_root()
    destination = root / "outputs" / "REPORT.md" if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "REPORT.md", destination)
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or reproduce sparse-circuit ablation and progressive PLOT experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("download-models")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--cuda", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--out", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), required=True)
    run.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.command == "audit":
        print(json.dumps(audit(), indent=2, sort_keys=True))
    elif args.command == "download-models":
        download_models()
    elif args.command == "smoke":
        from .smoke import run as run_smoke

        run_smoke(cuda=args.cuda)
    elif args.command == "report":
        _report(args.out)
    elif args.experiment == "all":
        for name in EXPERIMENTS:
            _run_experiment(name, cuda=args.cuda)
    else:
        _run_experiment(args.experiment, cuda=args.cuda)
