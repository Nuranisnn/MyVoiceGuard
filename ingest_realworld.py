"""
Copy clips into DATASET_REALWORLD so train_model.py gives them extra sample weight.

Usage (from project root):
  python ingest_realworld.py path/to/clip.mp3 REAL NAJIB
  python ingest_realworld.py path/to/clip.wav FAKE MAHATHIR

Speakers: ANWAR, NAJIB, MAHATHIR
Labels: REAL, FAKE
"""

import os
import shutil
import sys
import uuid


ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_SPEAKERS = frozenset({"ANWAR", "NAJIB", "MAHATHIR"})
ALLOWED_LABELS = frozenset({"REAL", "FAKE"})


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    src = os.path.abspath(sys.argv[1])
    label = sys.argv[2].upper()
    speaker = sys.argv[3].upper()
    if label not in ALLOWED_LABELS:
        print("Label must be REAL or FAKE.")
        sys.exit(1)
    if speaker not in ALLOWED_SPEAKERS:
        print("Speaker must be one of: ANWAR, NAJIB, MAHATHIR.")
        sys.exit(1)
    if not os.path.isfile(src):
        print(f"Not a file: {src}")
        sys.exit(1)
    dest_dir = os.path.join(ROOT, "DATASET_REALWORLD", label, speaker)
    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(src)[1] or ".wav"
    dest = os.path.join(dest_dir, f"ingest_{uuid.uuid4().hex[:12]}{ext}")
    shutil.copy2(src, dest)
    print(f"Copied -> {dest}")
    print("Next: python train_model.py")


if __name__ == "__main__":
    main()
