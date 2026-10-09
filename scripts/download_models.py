"""Download every model Kin needs into ./models and print their sizes.

Same as `uv run kin download-models`; pass --fallback for the GGUF LLM, --router for Gemma 3 270M.
"""

import sys

from kin.cli import main

if __name__ == "__main__":
    sys.exit(main(["download-models", *sys.argv[1:]]))
