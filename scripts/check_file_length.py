"""Pre-commit hook: fail on Python files longer than the structure limit.

Library code (packages/*/src) is capped tighter than tests and scripts. A file
that hits the cap should be split by responsibility, not squeezed under it.

    uv run python scripts/check_file_length.py <files...>
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_MAX_LINES = 400
OTHER_MAX_LINES = 500


def limit_for(path: Path) -> int:
    parts = path.as_posix().split("/")
    is_src = len(parts) > 2 and parts[0] == "packages" and parts[2] == "src"
    return SRC_MAX_LINES if is_src else OTHER_MAX_LINES


def main(argv: list[str]) -> int:
    failed = False
    for name in argv:
        path = Path(name)
        n = len(path.read_text(encoding="utf-8").splitlines())
        if n > (limit := limit_for(path)):
            print(f"{path.as_posix()}: {n} lines > {limit}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
