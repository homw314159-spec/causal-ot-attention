"""Language-model training entry point for the paper reproduction."""

from __future__ import annotations

import argparse

import yaml

from geoattn_train.config import load_config, to_dict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="geoattn-train")
    parser.add_argument("--config", required=True, help="path to a language-model YAML")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=3e-4")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved configuration")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if cfg.task != "lm":
        raise SystemExit("This paper release supports task=lm.")
    if args.dry_run:
        print(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        return
    from geoattn_train.lm_trainer import train_lm

    print(f"run saved to {train_lm(cfg)}")


if __name__ == "__main__":
    main()
