"""
MyVoiceGuard - train_model.py
Place this file in: MyVoiceGuardWebsite/Backend/train_model.py
Run from ANYWHERE:  python Backend/train_model.py
                OR: cd Backend && python train_model.py
"""

import os, sys, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# PATHS
# Supports both locations:
#   1) MyVoiceGuardWebsite/Backend/train_model.py
#   2) MyVoiceGuardWebsite/train_model.py
# =============================================================
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(BACKEND_DIR, "DATASET")):
    ROOT_DIR = BACKEND_DIR
else:
    ROOT_DIR = os.path.dirname(BACKEND_DIR)

# Try both DATASET and dataset (case-insensitive search)
DATASET_DIR = None
for name in ["DATASET", "dataset", "Dataset"]:
    candidate = os.path.join(ROOT_DIR, name)
    if os.path.isdir(candidate):
        DATASET_DIR = candidate
        break

if DATASET_DIR is None:
    # Last resort: show what IS in root so user can see
    print(f"\n[ERROR] Cannot find DATASET folder in: {ROOT_DIR}")
    print(f"   Contents of {ROOT_DIR}:")
    for item in os.listdir(ROOT_DIR):
        print(f"     {item}")
    sys.exit(1)

# Find real/ and fake/ subfolders (case-insensitive)
def find_subfolder(parent, names):
    for name in names:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    # also search by lowercased listdir
    try:
        for item in os.listdir(parent):
            if item.lower() in [n.lower() for n in names]:
                return os.path.join(parent, item)
    except Exception:
        pass
    return None

REAL_DIR = find_subfolder(DATASET_DIR, ["real", "REAL", "Real"])
FAKE_DIR = find_subfolder(DATASET_DIR, ["fake", "FAKE", "Fake"])

REALWORLD_DIR = None
for name in ["DATASET_REALWORLD", "dataset_realworld", "Dataset_RealWorld"]:
    candidate = os.path.join(ROOT_DIR, name)
    if os.path.isdir(candidate):
        REALWORLD_DIR = candidate
        break

REALWORLD_REAL_DIR = find_subfolder(REALWORLD_DIR, ["real", "REAL", "Real"]) if REALWORLD_DIR else None
REALWORLD_FAKE_DIR = find_subfolder(REALWORLD_DIR, ["fake", "FAKE", "Fake"]) if REALWORLD_DIR else None

_backend_dir = os.path.join(ROOT_DIR, "Backend")
if os.path.isdir(_backend_dir):
    MODEL_DIR = _backend_dir
else:
    MODEL_DIR = BACKEND_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.txt")

print("\n" + "="*65)
print("  MyVoiceGuard - Training Script")
print("="*65)
print(f"  Backend : {BACKEND_DIR}")
print(f"  Root    : {ROOT_DIR}")
print(f"  Dataset : {DATASET_DIR}")
print(f"  Real    : {REAL_DIR}")
print(f"  Fake    : {FAKE_DIR}")
print(f"  RealWld : {REALWORLD_DIR if REALWORLD_DIR else '(not found - optional)'}")
print(f"  Model   : {MODEL_PATH}")
print("="*65)

if not REAL_DIR:
    print(f"\n[ERROR] No 'real' or 'REAL' folder found inside {DATASET_DIR}")
    print(f"   Folders found: {os.listdir(DATASET_DIR)}")
    sys.exit(1)

if not FAKE_DIR:
    print(f"\n[ERROR] No 'fake' or 'FAKE' folder found inside {DATASET_DIR}")
    print(f"   Folders found: {os.listdir(DATASET_DIR)}")
    sys.exit(1)

# =============================================================
# CHECK DEPENDENCIES
# =============================================================
def check_deps():
    missing = []
    for pkg, imp in [("librosa","librosa"), ("scikit-learn","sklearn"),
                     ("joblib","joblib"), ("numpy","numpy"),
                     ("pydub","pydub"), ("soundfile","soundfile")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing packages: {missing}")
        print(f"   Run: pip install {' '.join(missing)}")
        sys.exit(1)
    print("\n[OK] All dependencies OK")

check_deps()

import librosa
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib

# =============================================================
# COLLECT AUDIO FILES RECURSIVELY
# Handles: .wav .WAV .mp3 .MP3 .ogg .OGG .m4a .M4A etc.
# =============================================================
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm"}

def collect_audio_files(root_folder):
    found = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXT:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)

# =============================================================
# AUDIO TO WAV CONVERSION
# =============================================================
def to_wav(src, dst):
    try:
        audio = AudioSegment.from_file(src)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(dst, format="wav")
        return True
    except Exception as e1:
        try:
            import soundfile as sf
            y, _ = librosa.load(src, sr=16000, mono=True)
            sf.write(dst, y, 16000)
            return True
        except Exception as e2:
            print(f"\n     [WARN] Convert failed: {e1} | {e2}")
            return False

# =============================================================
# FEATURE EXTRACTION  (44-dim - identical to app.py)
# =============================================================
N_MFCC = 13

def extract_features_from_audio(y, sr=16000):
    """44-dim MFCC-style vector (same as app.py). Uses at most 30s of the passed buffer."""
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < 1600:
        y = np.pad(y, (0, int(1600 - y.size)))
    max_samples = int(30 * sr)
    if y.size > max_samples:
        y = y[:max_samples].copy()

    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean  = np.mean(mfcc, axis=1)
    mfcc_std   = np.std(mfcc,  axis=1)
    delta      = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    cent  = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr   = librosa.feature.zero_crossing_rate(y)
    rms   = librosa.feature.rms(y=y)
    extra = np.array([
        np.mean(cent), np.std(cent),
        np.mean(zcr),
        np.mean(rms), np.std(rms),
    ])
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, extra])


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, duration=30, mono=True)
    return extract_features_from_audio(y, sr)


def _realworld_segment_params():
    """Overlapping windows so each long YouTube file contributes many training points."""
    try:
        w = float(os.environ.get("MV_RW_SEGMENT_SEC", "18").strip() or "18")
    except Exception:
        w = 18.0
    try:
        h = float(os.environ.get("MV_RW_HOP_SEC", "9").strip() or "9")
    except Exception:
        h = 9.0
    try:
        mx = int(os.environ.get("MV_RW_MAX_SEGMENTS", "28").strip() or "28")
    except Exception:
        mx = 28
    w = max(6.0, min(28.0, w))
    h = max(2.0, min(w - 0.5, h))
    mx = max(1, min(80, mx))
    return w, h, mx


def load_class_realworld(folder, label, domain_name="realworld"):
    """
    Like load_class(), but each audio file yields multiple feature rows using
    sliding windows (critical for long YouTube clips vs one 30s vector).
    """
    label_name = "REAL" if label == 1 else "FAKE"
    all_files = collect_audio_files(folder)
    if not all_files:
        print(f"\n  [WARN] [{label_name}] No audio files under: {folder}")
        return np.array([]), np.array([]), np.array([])

    window_sec, hop_sec, max_seg = _realworld_segment_params()
    win = int(window_sec * 16000)
    hop = int(hop_sec * 16000)
    print(
        f"\n  [{label_name}] {len(all_files)} file(s)  "
        f"(multi-segment: win={window_sec:.1f}s hop={hop_sec:.1}s max_seg={max_seg})"
    )

    tmp_dir = os.path.join(BACKEND_DIR, "_tmp_wav_")
    os.makedirs(tmp_dir, exist_ok=True)
    X, y, domain = [], [], []
    ok_files = skip = total_seg = 0

    for i, src in enumerate(all_files):
        dst = os.path.join(tmp_dir, f"rw_{label}_{i}.wav")
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    [{label_name}] file {i+1}/{len(all_files)} ...", end="\r", flush=True)
        if not to_wav(src, dst):
            skip += 1
            continue
        try:
            y_full, _ = librosa.load(dst, sr=16000, mono=True)
        except Exception:
            skip += 1
            if os.path.exists(dst):
                os.remove(dst)
            continue
        n_seg = 0
        if len(y_full) <= win:
            X.append(extract_features_from_audio(y_full))
            y.append(label)
            domain.append(domain_name)
            n_seg = 1
        else:
            last_start = len(y_full) - win
            starts = list(range(0, last_start + 1, hop))
            if starts[-1] < last_start:
                starts.append(last_start)
            for start in starts[:max_seg]:
                chunk = y_full[start : start + win]
                X.append(extract_features_from_audio(chunk))
                y.append(label)
                domain.append(domain_name)
                n_seg += 1
        total_seg += n_seg
        ok_files += 1
        try:
            if os.path.exists(dst):
                os.remove(dst)
        except Exception:
            pass

    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(
        f"\n  [{label_name}] [OK] {ok_files} file(s) -> {total_seg} feature rows  "
        f"[WARN] {skip} file(s) skipped"
    )
    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.array(X), np.array(y), np.array(domain)

# =============================================================
# LOAD ONE CLASS (recursively, with progress)
# label: 1 = REAL,  0 = FAKE
# =============================================================
def load_class(folder, label, domain_name):
    label_name = "REAL" if label == 1 else "FAKE"
    all_files  = collect_audio_files(folder)

    if not all_files:
        print(f"\n  [WARN] [{label_name}] No audio files found under: {folder}")
        print(f"        Subfolders found: {os.listdir(folder)}")
        # Show what's in each subfolder
        for sub in os.listdir(folder):
            subpath = os.path.join(folder, sub)
            if os.path.isdir(subpath):
                contents = os.listdir(subpath)
                print(f"          {sub}/ -> {len(contents)} items, e.g. {contents[:3]}")
        return np.array([]), np.array([]), np.array([])

    # Show breakdown by subfolder
    print(f"\n  [{label_name}] {len(all_files)} file(s) found")
    subfolders = {}
    for fp in all_files:
        sub = os.path.basename(os.path.dirname(fp))
        subfolders[sub] = subfolders.get(sub, 0) + 1
    for sub, count in subfolders.items():
        print(f"    {sub}/  ->  {count} file(s)")

    tmp_dir = os.path.join(BACKEND_DIR, "_tmp_wav_")
    os.makedirs(tmp_dir, exist_ok=True)

    X, y, domain = [], [], []
    ok = skip = 0

    for i, src in enumerate(all_files):
        dst = os.path.join(tmp_dir, f"tmp_{label}_{i}.wav")
        if (i+1) % 20 == 0 or i == 0:
            print(f"    [{label_name}] {i+1}/{len(all_files)} processed...", end="\r", flush=True)
        if not to_wav(src, dst):
            skip += 1; continue
        try:
            feat = extract_features(dst)
            X.append(feat); y.append(label); domain.append(domain_name); ok += 1
        except Exception:
            skip += 1
        finally:
            if os.path.exists(dst):
                os.remove(dst)

    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(f"\n  [{label_name}] [OK] {ok} extracted   [WARN] {skip} skipped")
    return np.array(X), np.array(y), np.array(domain)


def evaluate_domain(model, X, y, domain_mask, tag):
    if np.sum(domain_mask) == 0:
        return None
    Xd = X[domain_mask]
    yd = y[domain_mask]
    yp = model.predict(Xd)
    acc = accuracy_score(yd, yp)
    f1 = f1_score(yd, yp, average="macro", zero_division=0)
    cm = confusion_matrix(yd, yp)
    print(f"   {tag} -> Accuracy={acc:.4f} F1={f1:.4f} Samples={len(yd)}")
    return f"{tag}: Accuracy={acc:.4f} F1={f1:.4f} Samples={len(yd)} Confusion={cm.tolist()}"


def compute_sample_weights(domain):
    """
    Base clips use weight 1.0. DATASET_REALWORLD clips get larger weights so a
    small number of YouTube-style files actually move the decision boundary.

    Target: real-world rows account for ~MV_REALWORLD_TARGET_MASS of total
    sample weight (default 0.22). Solves:
        rw_n * w_rw = mass * (base_n + rw_n * w_rw)
        => w_rw = mass * base_n / (rw_n * (1 - mass))

    Override with env MV_REALWORLD_TARGET_MASS (e.g. 0.15 .. 0.35).
    """
    domain = np.asarray(domain, dtype=object)
    rw_mask = domain == "realworld"
    rw_n = int(np.sum(rw_mask))
    if rw_n == 0:
        return np.ones(len(domain), dtype=np.float64), ""
    base_n = int(np.sum(~rw_mask))
    raw = os.environ.get("MV_REALWORLD_TARGET_MASS", "0.22").strip()
    try:
        mass = float(raw)
    except ValueError:
        mass = 0.22
    mass = min(0.45, max(0.06, mass))
    denom = float(rw_n) * max(1e-9, (1.0 - mass))
    w_rw_uncapped = (mass * float(base_n)) / denom
    w_rw = min(120.0, max(3.0, w_rw_uncapped))
    w = np.where(rw_mask, w_rw, 1.0).astype(np.float64)
    total = float(np.sum(w))
    rw_share = float(np.sum(w[rw_mask])) / total
    print(
        f"\n[INFO] Sample weights: base=1.0  realworld_per_clip={w_rw:.2f}  "
        f"(clips: {rw_n} rw / {base_n} base)"
    )
    print(f"        Real-world share of total weight: {rw_share*100:.1f}%  (target mass={mass*100:.0f}%)")
    if w_rw + 1e-6 < w_rw_uncapped:
        print(f"        (weight capped from {w_rw_uncapped:.1f}; raise cap in code or add more rw clips)")
    info = (
        f"realworld_weight_per_clip={w_rw:.2f} rw_clips={rw_n} base_clips={base_n} "
        f"rw_weight_share={rw_share*100:.1f}% target_mass={mass*100:.0f}%"
    )
    return w, info


# =============================================================
# TRAIN
# =============================================================
def train():
    print("\n[INFO] Scanning dataset folders...")

    real_count = len(collect_audio_files(REAL_DIR))
    fake_count = len(collect_audio_files(FAKE_DIR))
    print(f"  REAL folder: {real_count} file(s)")
    print(f"  FAKE folder: {fake_count} file(s)")

    if real_count == 0:
        print(f"\n[ERROR] No audio files found in REAL folder: {REAL_DIR}")
        print("   Subfolders inside REAL:")
        for item in os.listdir(REAL_DIR):
            itempath = os.path.join(REAL_DIR, item)
            if os.path.isdir(itempath):
                files = os.listdir(itempath)
                print(f"     {item}/  ->  {len(files)} items  (e.g. {files[:2]})")
        return

    if fake_count == 0:
        print(f"\n[ERROR] No audio files found in FAKE folder: {FAKE_DIR}")
        return

    print("\n[INFO] Extracting REAL features...")
    X_real, y_real, d_real = load_class(REAL_DIR, label=1, domain_name="base")

    print("\n[INFO] Extracting FAKE features...")
    X_fake, y_fake, d_fake = load_class(FAKE_DIR, label=0, domain_name="base")

    if len(X_real) == 0 or len(X_fake) == 0:
        print("\n[ERROR] Feature extraction failed. Check FFmpeg: ffmpeg -version")
        return

    X_parts = [X_real, X_fake]
    y_parts = [y_real, y_fake]
    d_parts = [d_real, d_fake]

    # Optional real-world calibration set
    rw_real_count = rw_fake_count = 0
    if REALWORLD_REAL_DIR and REALWORLD_FAKE_DIR:
        rw_real_count = len(collect_audio_files(REALWORLD_REAL_DIR))
        rw_fake_count = len(collect_audio_files(REALWORLD_FAKE_DIR))
        print(f"\n[INFO] Real-world calibration files detected:")
        print(f"   REALWORLD REAL: {rw_real_count}")
        print(f"   REALWORLD FAKE: {rw_fake_count}")
        if rw_real_count > 0 and rw_fake_count > 0:
            print("\n[INFO] Extracting REALWORLD REAL features (multi-segment per file)...")
            X_rw_real, y_rw_real, d_rw_real = load_class_realworld(
                REALWORLD_REAL_DIR, label=1, domain_name="realworld"
            )
            print("\n[INFO] Extracting REALWORLD FAKE features (multi-segment per file)...")
            X_rw_fake, y_rw_fake, d_rw_fake = load_class_realworld(
                REALWORLD_FAKE_DIR, label=0, domain_name="realworld"
            )
            if len(X_rw_real) > 0 and len(X_rw_fake) > 0:
                X_parts += [X_rw_real, X_rw_fake]
                y_parts += [y_rw_real, y_rw_fake]
                d_parts += [d_rw_real, d_rw_fake]
            else:
                print("[WARN] Skipping real-world set because one side extracted zero features.")
        else:
            print("[WARN] Real-world folder exists but both REAL and FAKE are required.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    domain = np.concatenate(d_parts)
    n_real = int(np.sum(y == 1))
    n_fake = int(np.sum(y == 0))
    n_base = int(np.sum(domain == "base"))
    n_rw = int(np.sum(domain == "realworld"))

    print(f"\n[INFO] Dataset ready:")
    print(f"   REAL(1) : {n_real} samples")
    print(f"   FAKE(0) : {n_fake} samples")
    print(f"   Base samples     : {n_base}")
    print(f"   Real-world samples: {n_rw}")
    print(f"   Features: {X.shape[1]} per sample")

    # Build pipeline — when real-world rows exist, use a shallower RF so the
    # boundary generalizes off the 1200-sample memorized manifold.
    if n_rw > 0:
        print(
            "\n[INFO] Real-world samples present -> regularized RandomForest "
            "(better OOD / YouTube-style audio)."
        )
        clf = RandomForestClassifier(
            n_estimators      = 280,
            max_depth         = 14,
            min_samples_split = 6,
            min_samples_leaf  = 4,
            max_features      = "sqrt",
            class_weight      = "balanced_subsample",
            random_state      = 42,
            n_jobs            = -1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators      = 400,
            max_depth         = 22,
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            class_weight      = "balanced_subsample",
            random_state      = 42,
            n_jobs            = -1,
        )
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    report_lines = [
        "MyVoiceGuard Training Report", "="*50,
        f"REAL : {n_real}   FAKE : {n_fake}",
        f"Base samples={n_base}   Real-world samples={n_rw}",
        f"Features: {X.shape[1]}   Labels: 0=FAKE  1=REAL",
        "",
    ]

    # Cross-validation
    if n_real >= 3 and n_fake >= 3 and len(y) >= 6:
        n_splits = min(5, min(n_real, n_fake))
        print(f"\n[INFO] {n_splits}-fold cross-validation...")
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_acc = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_f1  = cross_val_score(pipeline, X, y, cv=skf, scoring="f1",       n_jobs=-1)
        print(f"   Accuracy : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
        print(f"   F1-score : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
        report_lines += [
            f"{n_splits}-fold CV: Accuracy={cv_acc.mean():.4f} F1={cv_f1.mean():.4f}", ""
        ]

    # Hold-out validation (realistic estimate for deployment behavior)
    print("\n[INFO] Running hold-out validation split (80/20)...")
    X_train, X_val, y_train, y_val, d_train, d_val = train_test_split(
        X, y, domain, test_size=0.2, random_state=42, stratify=y
    )
    train_weights, _winfo = compute_sample_weights(d_train)
    pipeline.fit(X_train, y_train, clf__sample_weight=train_weights)
    y_val_pred = pipeline.predict(X_val)
    classes_val = list(pipeline.named_steps["clf"].classes_)
    real_idx_val = classes_val.index(1) if 1 in classes_val else 0
    y_val_prob = pipeline.predict_proba(X_val)[:, real_idx_val]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    val_cr = classification_report(y_val, y_val_pred, target_names=["FAKE(0)", "REAL(1)"])

    print(f"   Validation Accuracy : {val_acc:.4f}")
    print(f"   Validation F1       : {val_f1:.4f}")
    print(f"   Validation Precision: {val_prec:.4f}")
    print(f"   Validation Recall   : {val_rec:.4f}")
    print(f"   Validation Confusion:\n{val_cm}")
    report_lines += [
        "Validation (80/20 split):",
        f"Accuracy={val_acc:.4f} F1={val_f1:.4f} Precision={val_prec:.4f} Recall={val_rec:.4f}",
        f"Confusion:\n{val_cm}",
        val_cr,
        "",
    ]
    print("   Validation by domain:")
    base_val_line = evaluate_domain(pipeline, X_val, y_val, d_val == "base", "Base validation")
    rw_val_line = evaluate_domain(pipeline, X_val, y_val, d_val == "realworld", "Real-world validation")
    if base_val_line:
        report_lines.append(base_val_line)
    if rw_val_line:
        report_lines.append(rw_val_line)
    try:
        val_auc = roc_auc_score(y_val, y_val_prob)
        print(f"   Validation ROC-AUC  : {val_auc:.4f}")
        report_lines.append(f"Validation ROC-AUC={val_auc:.4f}")
    except Exception:
        pass

    # Final fit on full dataset for exported model
    print("\n[INFO] Training final model on full dataset...")
    t0 = time.time()
    final_weights, weight_info = compute_sample_weights(domain)
    pipeline.fit(X, y, clf__sample_weight=final_weights)
    elapsed = time.time() - t0
    if weight_info:
        report_lines.append(f"Sample weights (full data): {weight_info}")

    clf     = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    print(f"\n[OK] Model classes: {classes}  -> 0=FAKE  1=REAL")

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, classes.index(1)]
    cr = classification_report(y, y_pred, target_names=["FAKE(0)", "REAL(1)"])
    cm = confusion_matrix(y, y_pred)
    print("\n[INFO] In-sample Report (diagnostic only):\n" + cr)
    print(f"In-sample Confusion Matrix:\n{cm}")

    try:
        auc = roc_auc_score(y, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
        report_lines.append(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    report_lines += [cr, f"In-sample confusion:\n{cm}", f"Time: {elapsed:.1f}s"]

    # Save
    joblib.dump(pipeline, MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*65}")
    print(f"  [OK] model.pkl saved  ->  {MODEL_PATH}")
    print(f"  [OK] training_report.txt saved ->  {REPORT_PATH}")
    print(f"\n  NEXT:")
    print(f"  1. cd to MyVoiceGuardWebsite root")
    print(f"  2. python app.py")
    print(f"  3. http://127.0.0.1:5000/health")
    print(f'     Must show: "model_classes": ["0","1"]')
    print(f"{'='*65}\n")


if __name__ == "__main__":
    train()