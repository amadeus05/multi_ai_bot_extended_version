from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
src_path = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_path]
sys.path.insert(0, src_path)
