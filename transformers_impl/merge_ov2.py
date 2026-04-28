"""Backward-compat shim: forwards to `transformers_impl.merge_ov2.cli` with --variant dense."""

import sys

from transformers_impl.merge_ov2.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    sys.exit(main(["merge", "--variant", "dense", *argv]))
