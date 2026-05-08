MyVoiceGuard — optional “real-world” training clips
===================================================

Drop YouTube exports (or any extra MP3/WAV/OGG/M4A) here so they are merged during
training with higher sample weight than the main DATASET folder.

Layout:
  DATASET_REALWORLD/REAL/<ANWAR|NAJIB|MAHATHIR>/*.wav
  DATASET_REALWORLD/FAKE/<ANWAR|NAJIB|MAHATHIR>/*.wav

Or use from the project root:
  python ingest_realworld.py "C:\path\to\najib_speech.mp3" REAL NAJIB

Then:
  python train_model.py
  (writes Backend\model.pkl when the Backend folder exists)

Training gives DATASET_REALWORLD clips much higher sample weight than before
(few YouTube files can actually steer the model). Optional tuning:
  set MV_REALWORLD_TARGET_MASS=0.28
  (fraction of total tree sample weight aimed at real-world; default 0.22, max ~0.45)

Each long real-world file is scanned with sliding windows (not just the first 30s):
  MV_RW_SEGMENT_SEC  (default 18)   MV_RW_HOP_SEC  (default 9)
  MV_RW_MAX_SEGMENTS (default 28)

Inference averages the same style of windows (defaults 18s / 9s hop / 14 segments):
  MV_INFER_SEGMENT_SEC  MV_INFER_HOP_SEC  MV_INFER_MAX_SEGMENTS

Check layout, counts, model, and reference WAVs anytime:
  python diagnose.py
  python diagnose.py --refs-cosine
  python diagnose.py --strict

Same JSON from the API (with app running):
  GET http://127.0.0.1:5000/health?diagnostics=1
  GET http://127.0.0.1:5000/health?diagnostics=full
  Set ENABLE_HEALTH_DIAGNOSTICS=0 on the server to turn this off.
