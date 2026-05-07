"""Backward-compatible script entrypoint for the sustainability experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sustainability.cli import main

if __name__ == "__main__":
    main()
