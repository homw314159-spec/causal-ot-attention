"""Print or run paper training recipes, without cloud provisioning or shutdown actions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def commands(spec: dict, stage: str, arms: list[str] | None = None) -> list[list[str]]:
    section = spec[stage]
    chosen = arms or list(section["arms"])
    unknown = set(chosen) - set(section["arms"])
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    return [
        [
            sys.executable,
            "-m",
            "geoattn_train.cli",
            "--config",
            section["config"],
            *section["arms"][arm],
            *section.get("overrides", []),
            f"seed={seed}",
            f"name={section['prefix']}{arm}_s{seed}",
        ]
        for seed in section["seeds"]
        for arm in chosen
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", help="stage name in configs/reproduction.yaml")
    parser.add_argument("--arms", nargs="+", help="optional subset of the stage's arms")
    parser.add_argument("--run", action="store_true", help="execute instead of printing commands")
    parser.add_argument("--set", nargs="*", default=[], help="additional dotted overrides")
    args = parser.parse_args()
    spec = yaml.safe_load(Path("configs/reproduction.yaml").read_text(encoding="utf-8"))
    if args.stage not in spec:
        parser.error(f"Unknown stage; choose from {', '.join(spec)}")
    for command in commands(spec, args.stage, args.arms):
        command += args.set
        if args.run:
            subprocess.run(command, check=True)
        else:
            print(json.dumps(command))


if __name__ == "__main__":
    main()
