"""Enforce ADR-0001 layering for every workspace package.

Dependencies point one way: serve -> train -> kernels -> obs -> core. Each
package may import the standard library, itself, and exactly what ALLOWED lists.
A new package must be registered here before its tests pass.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]

ALLOWED: dict[str, set[str]] = {
    "geoattn_core": {"torch"},
    "geoattn_obs": {"torch", "geoattn_core"},
    "geoattn_kernels": {"torch", "triton", "geoattn_core"},
    "geoattn_train": {
        "torch",
        "torchvision",
        "yaml",
        "wandb",
        "numpy",
        "geoattn_core",
        "geoattn_obs",
        "geoattn_kernels",
    },
}


def _source_roots() -> dict[str, Path]:
    """Import name -> source dir, e.g. geoattn_core -> packages/geoattn-core/src/geoattn_core."""
    return {
        src.name: src
        for pkg in PACKAGES.iterdir()
        if (pkg / "pyproject.toml").is_file()
        for src in (pkg / "src").iterdir()
        if src.is_dir() and (src / "__init__.py").is_file()
    }


def _top_level_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_package_is_registered() -> None:
    assert set(_source_roots()) == set(ALLOWED)


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_imports_follow_layering(module: str) -> None:
    src = _source_roots()[module]
    allowed = ALLOWED[module] | {module}
    offenders: dict[str, set[str]] = {}
    for path in src.rglob("*.py"):
        bad = {
            name
            for name in _top_level_imports(path)
            if name not in sys.stdlib_module_names and name not in allowed
        }
        if bad:
            offenders[str(path.relative_to(src))] = bad
    assert not offenders, f"{module} imports outside its layer (ADR-0001): {offenders}"
