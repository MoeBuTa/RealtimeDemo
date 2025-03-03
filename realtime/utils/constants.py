from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1].resolve().parents[0]
CONFIG_FILE = PROJECT_DIR / "config.yml"

INPUT_DIR = PROJECT_DIR / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


