"""
Run training from the project root. When Backend/ exists, model.pkl is written there.

Usage:
  python retrain_and_deploy.py
"""

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "train_model.py")],
        cwd=ROOT,
        check=True,
    )
    print("\nRestart the API (e.g. python app.py) so it reloads Backend/model.pkl.")


if __name__ == "__main__":
    main()
