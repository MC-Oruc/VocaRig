"""Run VocaRig synthetic dataset generation from the repository checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocarig.synthetic.generate import main


if __name__ == "__main__":
    main()
