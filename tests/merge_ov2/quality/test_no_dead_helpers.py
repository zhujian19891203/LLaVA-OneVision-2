"""Phase 8 A10 (pytest entry): wrap check_no_dead_helpers.main() for `pytest`."""

from __future__ import annotations

from .check_no_dead_helpers import main


def test_no_dead_helpers() -> None:
    assert main() == 0
