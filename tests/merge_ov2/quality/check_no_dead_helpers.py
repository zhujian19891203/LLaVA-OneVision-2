"""Phase 8 A10: assert dead helpers and unsafe patterns are absent from new package.

Greps `transformers_impl/merge_ov2/**` for symbols that the refactor was
supposed to eliminate. Exits 1 with a list of violations, 0 if clean.

Forbidden symbols and rationale:
- `convert_block_to_rowmajor_layout`: legacy ViT block-order helper folded
  into the unified loader.
- `create_test_image`: legacy synthetic image helper, replaced by real-image
  validators.
- `numpy.dot`/`np.linalg.norm` cosine helper: replaced by torch-native cosine
  in shared `utils`.
- `model.load_state_dict(...strict=False)`: legacy lax-loading anti-pattern;
  the new loader uses strict in-place copy with explicit missing/extra reports.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PKG = Path(__file__).resolve().parents[3] / "transformers_impl" / "merge_ov2"

FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "convert_block_to_rowmajor_layout": re.compile(r"\bconvert_block_to_rowmajor_layout\b"),
    "create_test_image": re.compile(r"\bcreate_test_image\b"),
    "numpy_cosine_helper": re.compile(r"np\.dot\([^)]*\)\s*/\s*\(np\.linalg\.norm"),
    "lax_load_state_dict": re.compile(r"load_state_dict\([^)]*strict\s*=\s*False"),
}


def main() -> int:
    if not PKG.is_dir():
        print(f"FAIL: package not found at {PKG}", file=sys.stderr)
        return 1
    violations: list[tuple[str, str, int, str]] = []
    for py in sorted(PKG.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for name, pat in FORBIDDEN_PATTERNS.items():
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                line = text.splitlines()[line_no - 1].strip()
                violations.append((str(py.relative_to(PKG.parent.parent)), name, line_no, line))
    if violations:
        print("FAIL: dead-code/anti-pattern violations:", file=sys.stderr)
        for path, name, line_no, line in violations:
            print(f"  {path}:{line_no}  [{name}]  {line}", file=sys.stderr)
        return 1
    print(f"OK: {PKG} contains no forbidden symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
