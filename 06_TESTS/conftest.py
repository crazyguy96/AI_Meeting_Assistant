import sys
from pathlib import Path

APPLICATION_DIR = Path(__file__).resolve().parents[1] / "01_APPLICATION"
sys.path.insert(0, str(APPLICATION_DIR))
