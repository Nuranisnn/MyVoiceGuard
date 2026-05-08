#!/usr/bin/env python3
"""
MyVoiceGuard - dataset, reference, and model diagnostics.

  python diagnose.py
  python diagnose.py --json
  python diagnose.py --strict          # exit 1 on dataset / model / ref issues
  python diagnose.py --refs-cosine   # pairwise cosine on 44-dim features (needs librosa)

HTTP (Flask): same payload via GET /health?diagnostics=1 or /health?diagnostics=full
  (full includes refs_cosine). Disable with env ENABLE_HEALTH_DIAGNOSTICS=0.

Does not import train_model.py (that module exits if DATASET is missing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from collections import defaultdict
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))

AUDIO_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm"}

# Match app.py speaker keys for reference_voice/*.wav
EXPECTED_REF_KEYS = ("anwar", "najib", "mahathir")


def find_subfolder(parent: str, names: list[str]) -> str | None:
    for name in names:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    try:
        for item in os.listdir(parent):
            if item.lower() in [n.lower() for n in names]:
                return os.path.join(parent, item)
    except OSError:
        pass
    return None


def find_dataset_dir() -> str | None:
    for name in ("DATASET", "dataset", "Dataset"):
        candidate = os.path.join(ROOT, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def find_realworld_dir() -> str | None:
    for name in ("DATASET_REALWORLD", "dataset_realworld", "Dataset_RealWorld"):
        candidate = os.path.join(ROOT, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def collect_audio_files(root_folder: str) -> list[str]:
    found: list[str] = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXT:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def counts_by_subfolder(files: list[str], class_root: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    class_root = os.path.normpath(class_root)
    for fp in files:
        rel = os.path.relpath(os.path.dirname(fp), class_root)
        key = rel if rel != "." else "_root_"
        out[key] += 1
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


def ext_breakdown(files: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for fp in files:
        ext = os.path.splitext(fp)[1].lower() or "(none)"
        out[ext] += 1
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


def wav_duration_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return None


def scan_class_folder(label: str, folder: str | None) -> dict[str, Any]:
    if not folder:
        return {"label": label, "path": None, "ok": False, "files": 0, "error": "folder missing"}
    files = collect_audio_files(folder)
    return {
        "label": label,
        "path": folder,
        "ok": len(files) > 0,
        "files": len(files),
        "bytes": sum(os.path.getsize(f) for f in files if os.path.isfile(f)),
        "extensions": ext_breakdown(files),
        "by_subfolder": counts_by_subfolder(files, folder),
    }


def load_model_info() -> dict[str, Any]:
    paths = [
        os.path.join(ROOT, "Backend", "model.pkl"),
        os.path.join(ROOT, "model.pkl"),
    ]
    out: dict[str, Any] = {"loaded": False, "path": None, "classes": None, "n_features_in_": None, "error": None}
    try:
        import joblib
    except ImportError:
        out["error"] = "joblib not installed"
        return out

    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            model = joblib.load(path)
            classes_attr = getattr(model, "classes_", None)
            if classes_attr is None:
                classes: list[str] = []
            else:
                classes = [str(c) for c in list(classes_attr)]
            n = getattr(model, "n_features_in_", None)
            if n is None:
                steps = getattr(model, "named_steps", None)
                if isinstance(steps, dict):
                    for step in steps.values():
                        n = getattr(step, "n_features_in_", None)
                        if n is not None:
                            break
            out["loaded"] = True
            out["path"] = path
            out["classes"] = classes
            out["n_features_in_"] = int(n) if n is not None else None
            return out
        except Exception as e:
            out["error"] = str(e)
            return out
    out["error"] = "no model.pkl in Backend/ or project root"
    return out


def scan_references() -> dict[str, Any]:
    ref_dir = os.path.join(ROOT, "reference_voice")
    rows: list[dict[str, Any]] = []
    keys_found: set[str] = set()
    if not os.path.isdir(ref_dir):
        return {
            "dir": ref_dir,
            "ok": False,
            "files": [],
            "missing_expected": list(EXPECTED_REF_KEYS),
            "error": "reference_voice folder missing",
        }
    for fname in sorted(os.listdir(ref_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        key = os.path.splitext(fname)[0].strip().lower()
        path = os.path.join(ref_dir, fname)
        dur = wav_duration_seconds(path)
        rows.append(
            {
                "file": fname,
                "key": key,
                "seconds": round(dur, 2) if dur is not None else None,
                "bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
            }
        )
        if key:
            keys_found.add(key)
    missing = [k for k in EXPECTED_REF_KEYS if k not in keys_found]
    return {
        "dir": ref_dir,
        "ok": len(missing) == 0,
        "files": rows,
        "missing_expected": missing,
        "error": None,
    }


def extract_features_44(wav_path: str):
    """Same 44-dim vector as app.py / train_model.py (needs librosa)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, duration=30, mono=True)
    if len(y) < 1600:
        y = np.pad(y, (0, 1600 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    extra = np.array(
        [
            np.mean(cent),
            np.std(cent),
            np.mean(zcr),
            np.mean(rms),
            np.std(rms),
        ]
    )
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, extra])


def cosine_similarity(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def refs_cosine_matrix(ref_scan: dict[str, Any]) -> dict[str, Any] | None:
    files = ref_scan.get("files") or []
    wavs = [os.path.join(ref_scan["dir"], r["file"]) for r in files if r.get("file", "").lower().endswith(".wav")]
    if len(wavs) < 2:
        return None
    try:
        feats = {os.path.basename(p): extract_features_44(p) for p in wavs}
    except Exception as e:
        return {"error": str(e)}
    names = sorted(feats.keys())
    mat: dict[str, dict[str, float]] = {}
    for n1 in names:
        mat[n1] = {}
        for n2 in names:
            mat[n1][n2] = round(cosine_similarity(feats[n1], feats[n2]), 4)
    return {"keys": names, "cosine": mat}


def build_report() -> dict[str, Any]:
    ds_dir = find_dataset_dir()
    real_dir = find_subfolder(ds_dir, ["real", "REAL", "Real"]) if ds_dir else None
    fake_dir = find_subfolder(ds_dir, ["fake", "FAKE", "Fake"]) if ds_dir else None

    rw_dir = find_realworld_dir()
    rw_real = find_subfolder(rw_dir, ["real", "REAL", "Real"]) if rw_dir else None
    rw_fake = find_subfolder(rw_dir, ["fake", "FAKE", "Fake"]) if rw_dir else None

    real_scan = scan_class_folder("REAL", real_dir)
    fake_scan = scan_class_folder("FAKE", fake_dir)
    rw_real_scan = scan_class_folder("REALWORLD_REAL", rw_real)
    rw_fake_scan = scan_class_folder("REALWORLD_FAKE", rw_fake)

    rw_merge_ok = (
        rw_dir is not None
        and rw_real_scan["files"] > 0
        and rw_fake_scan["files"] > 0
    )

    report: dict[str, Any] = {
        "root": ROOT,
        "dataset_dir": ds_dir,
        "base": {
            "real": real_scan,
            "fake": fake_scan,
            "balanced": (
                real_scan["files"] > 0
                and fake_scan["files"] > 0
                and abs(real_scan["files"] - fake_scan["files"])
                <= max(50, int(0.35 * max(real_scan["files"], fake_scan["files"])))
            ),
        },
        "realworld": {
            "dir": rw_dir,
            "real": rw_real_scan,
            "fake": rw_fake_scan,
            "will_merge_in_training": rw_merge_ok,
        },
        "references": scan_references(),
        "model": load_model_info(),
    }
    return report


def build_diagnostics_payload(
    include_refs_cosine: bool = False,
    include_strict_issues: bool = True,
) -> dict[str, Any]:
    """
    JSON-serializable dict for /health?diagnostics=1 (and CLI reuse).
    include_refs_cosine: loads each reference WAV with librosa (slower).
    """
    report = build_report()
    out: dict[str, Any] = dict(report)
    if include_strict_issues:
        out["strict_issues"] = strict_issues(report)
    if include_refs_cosine and out.get("references", {}).get("dir"):
        out["refs_cosine"] = refs_cosine_matrix(out["references"])
    return out


def strict_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not report.get("dataset_dir"):
        issues.append("DATASET folder not found under project root.")
    else:
        if not report["base"]["real"]["ok"]:
            issues.append("DATASET REAL branch has no audio files.")
        if not report["base"]["fake"]["ok"]:
            issues.append("DATASET FAKE branch has no audio files.")
    m = report.get("model") or {}
    if not m.get("loaded"):
        issues.append(f"Model not loaded: {m.get('error', 'unknown')}")
    refs = report.get("references") or {}
    if refs.get("missing_expected"):
        issues.append(
            "Missing reference WAV(s) for: "
            + ", ".join(refs["missing_expected"])
            + " (expected under reference_voice/)"
        )
    rw = report.get("realworld") or {}
    if rw.get("dir") and not rw.get("will_merge_in_training"):
        issues.append(
            "DATASET_REALWORLD exists but training will skip it until BOTH "
            "REAL and FAKE sides have at least one audio file each."
        )
    if report["base"]["real"]["ok"] and report["base"]["fake"]["ok"] and not report["base"]["balanced"]:
        issues.append(
            "Base REAL vs FAKE counts are imbalanced (>35% gap). "
            "Consider balancing for more stable probabilities."
        )
    return issues


def print_human(report: dict[str, Any], cosine_block: dict[str, Any] | None) -> None:
    print("\n" + "=" * 66)
    print("  MyVoiceGuard - diagnose")
    print("=" * 66)
    print(f"  Root: {report['root']}\n")

    ds = report["dataset_dir"]
    if not ds:
        print("  [ERROR] No DATASET folder found.\n")
    else:
        print(f"  Dataset: {ds}")
        for side in ("real", "fake"):
            s = report["base"][side]
            print(f"    {side.upper()}: {s['files']} files  (ok={s['ok']})")
            if s.get("by_subfolder"):
                for sub, n in list(s["by_subfolder"].items())[:12]:
                    print(f"      {sub}/  ->  {n}")
                if len(s["by_subfolder"]) > 12:
                    print(f"      ... +{len(s['by_subfolder']) - 12} more subfolders")
            if s.get("extensions"):
                print(f"      extensions: {s['extensions']}")
        print(f"    roughly_balanced: {report['base']['balanced']}")

    rw = report["realworld"]
    print(f"\n  Real-world: {rw['dir'] or '(not present)'}")
    print(f"    REAL files: {rw['real']['files']}   FAKE files: {rw['fake']['files']}")
    print(f"    merges in training: {rw['will_merge_in_training']}")

    refs = report["references"]
    print(f"\n  References: {refs['dir']}")
    if refs.get("error"):
        print(f"    [ERROR] {refs['error']}")
    for row in refs.get("files", []):
        dur = row.get("seconds")
        dur_s = f"{dur}s" if dur is not None else "dur?"
        print(f"    - {row['file']}  ({dur_s}, {row['bytes']} bytes)")
    if refs.get("missing_expected"):
        print(f"    [WARN] Missing expected: {', '.join(refs['missing_expected'])}.wav")

    m = report["model"]
    print("\n  Model:")
    if m.get("loaded"):
        print(f"    OK: {m['path']}")
        print(f"    classes: {m.get('classes')}   n_features_in_: {m.get('n_features_in_')}")
    else:
        print(f"    [WARN] {m.get('error', 'not loaded')}")

    if cosine_block:
        print("\n  Reference pairwise cosine (same 44-dim features as inference):")
        if cosine_block.get("error"):
            print(f"    [ERROR] {cosine_block['error']}")
        else:
            names = cosine_block["keys"]
            mat = cosine_block["cosine"]
            colw = max(len(n) for n in names) + 2
            header = "".join(n.ljust(colw) for n in [""] + names)
            print("    " + header)
            for n1 in names:
                line = n1.ljust(colw) + "".join(str(mat[n1][n2]).ljust(colw) for n2 in names)
                print("    " + line)
            print(
                "    (High cosine between two *different* speakers => ambiguous speaker ID; "
                "low self row is a bug.)"
            )
            off_diag = []
            for n1 in names:
                for n2 in names:
                    if n1 != n2:
                        off_diag.append(mat[n1][n2])
            if off_diag and max(off_diag) >= 0.85:
                print(
                    f"    [WARN] Max off-diagonal cosine is {max(off_diag):.4f} "
                    "(refs look almost identical in MFCC space). Re-record reference_voice "
                    "clips with distinct speech, or longer diverse segments."
                )

    print("\n" + "=" * 66 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="MyVoiceGuard dataset / model diagnostics.")
    ap.add_argument("--json", action="store_true", help="Print one JSON object to stdout.")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if dataset/model/refs/realworld rules fail (see script doc).",
    )
    ap.add_argument(
        "--refs-cosine",
        action="store_true",
        help="Print pairwise cosine similarity between reference_voice/*.wav features.",
    )
    args = ap.parse_args()

    if args.json:
        out = build_diagnostics_payload(
            include_refs_cosine=args.refs_cosine,
            include_strict_issues=True,
        )
        print(json.dumps(out, indent=2, default=str))
        issues = list(out.get("strict_issues") or [])
    else:
        report = build_report()
        cosine_block = None
        if args.refs_cosine and report["references"].get("dir"):
            cosine_block = refs_cosine_matrix(report["references"])
        print_human(report, cosine_block)
        issues = strict_issues(report)

    if args.strict and issues:
        if not args.json:
            print("Strict mode failures:")
            for i in issues:
                print(f"  - {i}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
