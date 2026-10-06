import sys
from pathlib import Path

# Root entrypoint that adds 01_APPLICATION to sys.path and launches the Gradio UI.
APPLICATION_DIR = Path(__file__).resolve().parent / "01_APPLICATION"
sys.path.insert(0, str(APPLICATION_DIR))

from app.ui.gradio_app import launch_app

if __name__ == "__main__":
    launch_app()
