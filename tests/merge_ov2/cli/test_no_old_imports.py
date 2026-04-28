"""Phase 8: assert no live caller imports `merge_ov2_moe` directly.

Per plan §917-921: outside `tests/legacy/` and the shim itself, no module
should `from merge_ov2_moe import ...` or `import merge_ov2_moe` at the
bare-name level. Imports through the package (`transformers_impl.merge_ov2_moe`)
are allowed only as the BC-shim entry point.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "tests/legacy", "build", "dist"}
EXCLUDE_FILES = {"transformers_impl/merge_ov2_moe.py"}

BARE_IMPORT = re.compile(r"^\s*(from\s+merge_ov2_moe\b|import\s+merge_ov2_moe\b)", re.MULTILINE)


def _excluded(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    if rel in EXCLUDE_FILES:
        return True
    parts = rel.split("/")
    for i in range(len(parts)):
        if "/".join(parts[: i + 1]) in EXCLUDE_DIRS:
            return True
    return False


def test_no_bare_merge_ov2_moe_imports() -> None:
    violations: list[str] = []
    for py in REPO.rglob("*.py"):
        if _excluded(py):
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in BARE_IMPORT.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            violations.append(f"{py.relative_to(REPO)}:{line_no}  {m.group(0).strip()}")
    assert not violations, "live callers of merge_ov2_moe found:\n  " + "\n  ".join(violations)
