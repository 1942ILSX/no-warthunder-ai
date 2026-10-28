"""PyCharm entry point. Run with no arguments for doctor, or add `train` parameters."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nowarthunder_ai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
