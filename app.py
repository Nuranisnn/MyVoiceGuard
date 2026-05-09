import os

# Render free tier + librosa/numba JIT can OOM with multi-threaded BLAS. Set before numpy/scipy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
# Avoid heavy Numba JIT compilation on low-memory Render workers.
# Can be overridden with MV_ENABLE_NUMBA_JIT=1 on larger instances.
if os.environ.get("MV_ENABLE_NUMBA_JIT", "").strip().lower() not in ("1", "true", "yes", "on"):
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import numpy as np
import time
import uuid
import warnings
import tempfile
import re
import shutil
import traceback

from werkzeug.exceptions import HTTPException

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Accept, Authorization, X-Requested-With"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


# ------------------------------------------------------------------
# Global error handlers
#
# Problem observed in browser:
# - POST /predict-file returns 500 and browser reports:
#   "No 'Access-Control-Allow-Origin' header is present"
#
# Sometimes the response that reaches the client doesn't include CORS
# headers (e.g., unhandled exceptions / worker errors). This ensures
# every error response has CORS so frontend can read the JSON error.
# ------------------------------------------------------------------
def _add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Accept, Authorization, X-Requested-With"
    )
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


@app.before_request
def handle_preflight():
    """CORS preflight must include Allow-* headers (some proxies strip after_request on 204)."""
    if request.method == "OPTIONS":
        return _add_cors_headers(make_response("", 204))


@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    payload = {
        "error": getattr(e, "name", "HTTPException"),
        "code": getattr(e, "code", None),
        "details": getattr(e, "description", None),
    }
    resp = jsonify(payload)
    resp.status_code = int(getattr(e, "code", 500) or 500)
    return _add_cors_headers(resp)


@app.errorhandler(Exception)
def handle_unexpected_exception(e: Exception):
    # Keep trace only when explicitly enabled.
    debug_trace = os.environ.get("MV_DEBUG_TRACEBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    payload = {
        "error": "Internal Server Error",
        "details": str(e),
    }
    if debug_trace:
        payload["traceback"] = traceback.format_exc()
    resp = jsonify(payload)
    resp.status_code = 500
    return _add_cors_headers(resp)

# =========================
# CONFIG
# =========================
# Binary thresholds: >=90% REAL, <90% FAKE (all speakers).
THRESHOLD_REAL = 90
THRESHOLD_FAKE = 90
KNOWN_SPEAKER_MIN_SIMILARITY = 0.72
# Reference speaker match (no filename): min cosine / ambiguity gates below.
# Minimum cosine sim to accept best reference (generic filenames / YouTube).
SPEAKER_REF_MIN_SIMILARITY = 0.30
# Ambiguity: only if runner-up is this strong AND very close to winner -> unknown.
SPEAKER_REF_SECOND_AMBIGUOUS = 0.42
SPEAKER_REF_AMBIGUITY_MARGIN = 0.028
MAX_FILE_SIZE  = 50 * 1024 * 1024  # 50 MB

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
reference_folder = os.path.join(BASE_DIR, "reference_voice")
temp_folder      = os.path.join(tempfile.gettempdir(), "myvoiceguard_temp")
os.makedirs(reference_folder, exist_ok=True)
os.makedirs(temp_folder,      exist_ok=True)


def _infer_low_memory_mode():
    """
    Render free tier (~512MB) SIGKILLs workers during librosa/numba JIT on long audio.
    RENDER is set by Render.com; MV_LOW_MEMORY forces the same caps elsewhere.
    MV_HIGH_MEMORY=1 disables caps (e.g. paid instance).
    """
    if os.environ.get("MV_HIGH_MEMORY", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("MV_LOW_MEMORY", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("RENDER"))


def _infer_max_audio_seconds():
    """Max seconds loaded into RAM for /predict-* scoring (not reference_voice clips)."""
    raw = os.environ.get("MV_INFER_MAX_AUDIO_SEC", "").strip()
    if raw:
        try:
            return float(max(15.0, min(600.0, float(raw))))
        except ValueError:
            pass
    return 90.0 if _infer_low_memory_mode() else 300.0


def _infer_segment_defaults():
    """(segment_sec, hop_sec, max_segments) when MV_INFER_* env vars are unset."""
    if _infer_low_memory_mode():
        return 12.0, 6.0, 6
    return 18.0, 9.0, 18


def _ffmpeg_available():
    """MP3/M4A/WebM decode (pydub/librosa) needs ffmpeg on PATH — check /health."""
    return shutil.which("ffmpeg") is not None


@app.route("/", methods=["GET"])
def root():
    """Avoid bare-domain 404; helps verify the Render service URL."""
    return jsonify(
        {
            "service": "MyVoiceGuard API",
            "health": "/health",
            "routes": "/routes",
            "endpoints": ["POST /predict-file", "POST /predict-url", "POST /predict-live"],
            "ffmpeg_in_path": _ffmpeg_available(),
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
        }
    )


@app.route("/web", methods=["GET"])
def web_index():
    """Serve frontend from Flask to avoid Live Server auto-reload resets."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/web/<path:asset_path>", methods=["GET"])
def web_assets(asset_path):
    full = os.path.join(BASE_DIR, asset_path)
    if os.path.isfile(full):
        return send_from_directory(BASE_DIR, asset_path)
    return jsonify({"error": "Asset not found"}), 404


@app.route("/<path:asset_path>", methods=["GET"])
def root_assets(asset_path):
    """
    Serve frontend assets when using /web entrypoint.
    Example: /script.js, /Photo/background.jpg
    Existing API routes keep priority over this catch-all route.
    """
    full = os.path.join(BASE_DIR, asset_path)
    if os.path.isfile(full):
        return send_from_directory(BASE_DIR, asset_path)
    return jsonify({"error": "Not found"}), 404

# region agent log
def _agent_debug_log(hypothesis_id, location, message, data, run_id="pre-fix"):
    """NDJSON to debug-db268d.log (debug mode)."""
    import json
    import time

    # Disable by default to avoid Live Server auto-reload loops while app runs normally.
    # Enable only when explicitly debugging:
    #   set MVG_AGENT_LOG=1   (Windows CMD)  or  $env:MVG_AGENT_LOG='1' (PowerShell)
    if os.environ.get("MVG_AGENT_LOG", "0").strip() != "1":
        return

    try:
        payload = {
            "sessionId": "db268d",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
        }
        log_path = os.path.join(BASE_DIR, "debug-db268d.log")
        with open(log_path, "a", encoding="utf-8") as _lf:
            _lf.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# endregion

# =========================
# POLITICIAN NAME HINTS
# Maps keywords found in filenames / URLs → politician key
# =========================
POLITICIAN_KEYWORDS = {
    "anwar": [
        "anwar", "anwar ibrahim", "anwaribrahim", "pm anwar",
        "perdana menteri anwar", "pkr", "keadilan"
    ],
    "najib": [
        "najib", "najib razak", "najibs", "1mdb", "bn najib",
        "pekan", "umno najib"
    ],
    "mahathir": [
        "mahathir", "tun mahathir", "mahathir mohamad", "tun m",
        "pejuang", "langkawi mahathir", "dr mahathir"
    ],
}

KNOWN_POLITICIAN_KEYS = frozenset(POLITICIAN_KEYWORDS.keys())


def detect_politician_from_text(text: str):
    """
    Return politician key if any keyword is found in text (case-insensitive).
    Returns None if no match.
    """
    text_lower = text.lower()
    # Remove punctuation / special chars to improve matching
    text_clean = re.sub(r"[^a-z0-9\s]", " ", text_lower)

    for key, keywords in POLITICIAN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_clean:
                print(f"[HINT] Politician '{key}' detected from text keyword '{kw}'")
                return key

    # Fallback: match names from reference_voice/*.wav dynamically
    # Example: "muhyiddin speech" will match reference "muhyiddin.wav"
    try:
        for fname in os.listdir(reference_folder):
            if not fname.lower().endswith(".wav"):
                continue
            ref_key = os.path.splitext(fname)[0].strip().lower()
            if ref_key and ref_key in text_clean:
                print(f"[HINT] Speaker '{ref_key}' detected from reference filename")
                return ref_key
    except Exception:
        pass
    return None

# =========================
# LOAD MODEL
# =========================
model        = None
model_status = "not loaded"
REAL_CLASS_INDEX_OVERRIDE = None

for path in [
    os.path.join(BASE_DIR, "Backend", "model.pkl"),
    os.path.join(BASE_DIR, "model.pkl"),
]:
    try:
        import joblib
        model        = joblib.load(path)
        model_status = f"loaded from {path}"
        print(f"[OK] Model loaded: {path}")
        break
    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"[WARN] Error loading {path}: {e}")

if model is None:
    print("[WARN] Using dummy model (no model.pkl found)")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(np.random.rand(100, 13), np.array([0] * 50 + [1] * 50))
    model_status = "dummy"

# =========================
# OPTIONAL IMPORTS
# =========================
try:
    import librosa
    LIBROSA_OK = True
    print("[OK] librosa OK")
except ImportError:
    LIBROSA_OK = False
    print("[ERROR] librosa missing - pip install librosa")

try:
    from pydub import AudioSegment
    PYDUB_OK = True
    print("[OK] pydub OK")
except ImportError:
    PYDUB_OK = False
    print("[ERROR] pydub missing - pip install pydub")

try:
    import yt_dlp
    YTDLP_OK = True
    print("[OK] yt-dlp OK")
except ImportError:
    YTDLP_OK = False
    print("[WARN] yt-dlp missing (needed for YouTube/TikTok)")

try:
    import requests as req_lib
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


def _yt_dlp_youtube_extractor_args():
    """
    YouTube often needs non-web clients when no Node/Deno JS runtime is installed.
    See: https://github.com/yt-dlp/yt-dlp/wiki/EJS
    """
    return {"youtube": {"player_client": ["android", "web", "mweb"]}}


def _youtube_oembed_title(url: str) -> str:
    """Public title without full yt-dlp extraction (helps name_hint when metadata is flaky)."""
    if not REQUESTS_OK:
        return ""
    if "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
        return ""
    try:
        from urllib.parse import quote

        q = quote(url, safe="")
        r = req_lib.get(
            f"https://www.youtube.com/oembed?format=json&url={q}",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MyVoiceGuard/1.0)"},
        )
        if r.status_code == 200:
            t = (r.json() or {}).get("title") or ""
            return str(t).strip()
    except Exception as e:
        print(f"[URL] oEmbed title failed: {e}")
    return ""


def _wav_duration_seconds(path: str):
    try:
        import wave

        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return None


def _load_audio_mono_16k(path: str, offset_sec: float = 0.0, duration_sec=None):
    """
    Robust loader for inference paths.
    Prefer soundfile (no librosa/numba resample JIT), fallback to librosa.
    Returns (y_float32_mono, sr=16000).
    """
    target_sr = 16000
    off = float(max(0.0, offset_sec or 0.0))
    dur = None if duration_sec is None else float(max(0.0, duration_sec))

    # Fast path: soundfile read with frame slicing
    try:
        import soundfile as sf

        info = sf.info(path)
        in_sr = int(info.samplerate or target_sr)
        start_frame = int(off * in_sr)
        n_frames = -1 if dur is None else int(max(1, dur * in_sr))
        y, sr = sf.read(
            path,
            start=start_frame,
            frames=n_frames,
            dtype="float32",
            always_2d=True,
        )
        if y is None or len(y) == 0:
            return np.zeros(1600, dtype=np.float32), target_sr
        y = np.mean(y, axis=1).astype(np.float32, copy=False)
        sr = int(sr or in_sr or target_sr)
        if sr == target_sr:
            return y, target_sr

        # Lightweight linear resample to 16k (avoids scipy/librosa heavy paths)
        n_in = len(y)
        if n_in < 2:
            return np.zeros(1600, dtype=np.float32), target_sr
        n_out = int(max(1, round(n_in * target_sr / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=True, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=True, dtype=np.float32)
        y_rs = np.interp(x_new, x_old, y).astype(np.float32, copy=False)
        return y_rs, target_sr
    except Exception as e:
        print(f"[AUDIO] soundfile load failed: {e}")

    # Fallback: librosa
    try:
        y, _ = librosa.load(path, sr=target_sr, mono=True, offset=off, duration=dur)
        y = np.asarray(y, dtype=np.float32).flatten()
        if y.size == 0:
            y = np.zeros(1600, dtype=np.float32)
        return y, target_sr
    except Exception as e:
        print(f"[AUDIO] librosa fallback load failed: {e}")
        return np.zeros(1600, dtype=np.float32), target_sr


def _youtube_time_token_to_seconds(token: str) -> float:
    """Parse YouTube t= values: 75, 75s, 1m15s, 1h2m3s."""
    import re

    t = (token or "").strip().lower()
    if not t:
        return 0.0
    if t.isdigit():
        return min(float(t), 7200.0)
    sec = 0.0
    for m in re.finditer(r"(\d+)h", t):
        sec += int(m.group(1)) * 3600
    for m in re.finditer(r"(\d+)m", t):
        sec += int(m.group(1)) * 60
    for m in re.finditer(r"(\d+)s", t):
        sec += int(m.group(1))
    if sec == 0.0 and re.fullmatch(r"\d+m\d+", t):
        parts = t.split("m", 1)
        sec = int(parts[0]) * 60 + int(re.sub(r"\D", "", parts[1]) or 0)
    return float(min(max(sec, 0.0), 7200.0))


def _parse_youtube_start_seconds(url: str) -> float:
    """Seconds to skip (browser ?t= / &start= / #t=). Full video is still downloaded."""
    if "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
        return 0.0
    try:
        from urllib.parse import urlparse, parse_qs

        u = urlparse(url.replace("&amp;", "&"))
        qs = parse_qs(u.query)
        for key in ("t", "start", "time_continue"):
            if key in qs and qs[key][0]:
                return _youtube_time_token_to_seconds(qs[key][0])
        frag = u.fragment or ""
        if frag.startswith("t="):
            return _youtube_time_token_to_seconds(frag[2:])
    except Exception:
        pass
    return 0.0


# =========================
# AUDIO CONVERSION
# =========================
def convert_to_wav(input_path, output_path):
    print(f"[CONVERT] {input_path} -> {output_path}")
    ext = os.path.splitext(input_path)[1].lower()
    compressed = ext in (".mp3", ".m4a", ".aac", ".webm", ".ogg", ".opus", ".flac")

    # Prefer ffmpeg first for compressed audio when available (Render Python image often lacks it;
    # Docker image in this repo installs ffmpeg — see Dockerfile / render.yaml).
    if compressed and _ffmpeg_available():
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    output_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print("[CONVERT] ffmpeg (preferred for compressed) OK")
                return True
            print(f"[CONVERT] ffmpeg preferred path failed: {result.stderr[-400:]}")
        except Exception as e:
            print(f"[CONVERT] ffmpeg preferred error: {e}")

    if PYDUB_OK:
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            print(f"[CONVERT] pydub OK  ({len(audio)/1000:.1f}s)")
            return True
        except Exception as e:
            print(f"[CONVERT] pydub failed: {e}")

    if LIBROSA_OK:
        try:
            import soundfile as sf

            conv_raw = os.environ.get("MV_CONVERT_MAX_AUDIO_SEC", "").strip()
            if conv_raw:
                try:
                    cmax = float(max(10.0, min(600.0, float(conv_raw))))
                except ValueError:
                    cmax = None
            else:
                cmax = 120.0 if _infer_low_memory_mode() else None
            if cmax is not None:
                y, _ = librosa.load(input_path, sr=16000, mono=True, duration=cmax)
                print(f"[CONVERT] librosa decode capped at {cmax}s (low-memory host)")
            else:
                y, _ = librosa.load(input_path, sr=16000, mono=True)
            sf.write(output_path, y, 16000)
            print(f"[CONVERT] librosa OK ({len(y)/16000:.1f}s)")
            return True
        except Exception as e:
            print(f"[CONVERT] librosa failed: {e}")

    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print("[CONVERT] ffmpeg subprocess OK")
            return True
        print(f"[CONVERT] ffmpeg failed: {result.stderr[-300:]}")
    except Exception as e:
        print(f"[CONVERT] ffmpeg subprocess error: {e}")

    return False

# =========================
# FEATURE EXTRACTION
# Returns a rich 44-dim vector (matches train_model.py):
#   13 MFCC means + 13 MFCC stds + 13 delta means + 5 spectral extras
# =========================
def _features_from_audio_y(y, sr=16000):
    """44-dim vector from mono float audio (uses at most 30s)."""
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < 1600:
        y = np.pad(y, (0, int(1600 - y.size)))
    max_samples = int(30 * sr)
    if y.size > max_samples:
        y = y[:max_samples].copy()

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc,  axis=1)
    delta       = librosa.feature.delta(mfcc)
    delta_mean  = np.mean(delta, axis=1)
    cent        = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)
    extra       = np.array([
        np.mean(cent), np.std(cent),
        np.mean(zcr),
        np.mean(rms), np.std(rms),
    ])
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, extra])


def extract_features(wav_path):
    """First 30s only — used for short reference_voice clips and speaker cosine."""
    if not LIBROSA_OK:
        return np.zeros(44)
    try:
        y, sr = _load_audio_mono_16k(wav_path, duration_sec=30.0)
        feats = _features_from_audio_y(y, sr)
        print(f"[FEATURES] shape={feats.shape}  norm={np.linalg.norm(feats):.2f}")
        return feats
    except Exception as e:
        print(f"[FEATURES] Error: {e}")
        return np.zeros(44)


def _segment_audio_chunks(y_full, sr=16000):
    """Sliding windows (mono samples). Same defaults as extract_features_for_predict."""
    dw, dh, dm = _infer_segment_defaults()
    try:
        wsec = float(os.environ.get("MV_INFER_SEGMENT_SEC", str(dw)).strip() or str(dw))
    except Exception:
        wsec = dw
    try:
        hsec = float(os.environ.get("MV_INFER_HOP_SEC", str(dh)).strip() or str(dh))
    except Exception:
        hsec = dh
    try:
        max_seg = int(os.environ.get("MV_INFER_MAX_SEGMENTS", str(dm)).strip() or str(dm))
    except Exception:
        max_seg = dm
    wsec = max(6.0, min(28.0, wsec))
    hsec = max(2.0, min(wsec - 0.5, hsec))
    max_seg = max(1, min(48, max_seg))

    win = int(wsec * 16000)
    hop = int(hsec * 16000)
    if len(y_full) <= win:
        return [y_full.copy()]
    last_start = len(y_full) - win
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] < last_start:
        starts.append(last_start)
    return [y_full[s : s + win].copy() for s in starts[:max_seg]]


def extract_features_for_predict(wav_path, start_offset_sec=0.0):
    """
    Legacy path: mean 44-dim over windows (used only as fallback).
    Prefer model_predict_multisegment() for scoring uploads.
    """
    if not LIBROSA_OK:
        return np.zeros(44)
    try:
        off = float(max(0.0, start_offset_sec))
        max_dur = _infer_max_audio_seconds()
        y_full, sr = _load_audio_mono_16k(
            wav_path, offset_sec=off, duration_sec=max_dur
        )
        chunks = _segment_audio_chunks(y_full, sr)
        del y_full
        vecs = [_features_from_audio_y(c, sr) for c in chunks]
        out = np.mean(np.stack(vecs, axis=0), axis=0)
        print(
            f"[FEATURES] predict mean-features n={len(vecs)} "
            f"norm={np.linalg.norm(out):.2f}"
        )
        return out
    except Exception as e:
        print(f"[FEATURES] predict multi-segment failed ({e}); using first 30s.")
        return extract_features(wav_path)


def model_predict_multisegment(wav_path, start_offset_sec=0.0):
    """
    Run the RF on each audio window, then aggregate probabilities (better than
    averaging MFCCs then one predict). Returns dict for process_audio.
    start_offset_sec: skip first N seconds (YouTube ?t= / &start= speech start).
    """
    out = {
        "raw_prob_real": 0.5,
        "pred_label": None,
        "classes": [],
        "real_idx": 0,
        "features_for_speaker": np.zeros(44),
        "n_segments": 0,
        "vote_real_fraction": 0.0,
        "segment_mean_raw_real": 0.0,
    }
    if not LIBROSA_OK:
        return out
    try:
        off = float(max(0.0, start_offset_sec))
        max_dur = _infer_max_audio_seconds()
        if off > 0:
            print(f"[MODEL] librosa offset={off:.2f}s (YouTube start param / speech skip)")
        print(
            f"[MODEL] librosa load duration_cap={max_dur}s low_memory_mode="
            f"{_infer_low_memory_mode()}"
        )
        y_full, sr = _load_audio_mono_16k(
            wav_path, offset_sec=off, duration_sec=max_dur
        )
        chunks = _segment_audio_chunks(y_full, sr)
        del y_full
        feats44 = [_features_from_audio_y(c, sr) for c in chunks]
        out["features_for_speaker"] = np.mean(np.stack(feats44, axis=0), axis=0)
        out["n_segments"] = len(feats44)

        classes = list(getattr(model, "classes_", []))
        out["classes"] = classes
        _infer_real_class_from_references(classes)
        real_idx = _get_real_class_index(classes)
        out["real_idx"] = real_idx
        real_class = classes[real_idx]

        prob_rows = []
        preds = []
        raw_reals = []
        for v in feats44:
            vin = _adapt_features_for_model(v)
            pr = model.predict_proba([vin])[0]
            prob_rows.append(pr)
            preds.append(model.predict([vin])[0])
            raw_reals.append(float(pr[real_idx]))
        P = np.stack(prob_rows, axis=0)
        mean_p = np.mean(P, axis=0)
        # Robust combine: mean vs upper-mid quantile (one noisy segment won't drag REAL down)
        raw_mean = float(np.mean(raw_reals))
        raw_p60 = float(np.percentile(raw_reals, 60))
        raw_prob = max(raw_mean, 0.5 * raw_mean + 0.5 * raw_p60)
        raw_prob = float(min(0.999, max(0.001, raw_prob)))
        out["segment_mean_raw_real"] = raw_mean
        out["raw_prob_real"] = raw_prob

        pred_from_mean = mean_p.argmax()
        out["pred_label"] = classes[pred_from_mean]

        n_real_votes = sum(
            1 for p in preds if p == real_class or str(p) == str(real_class)
        )
        out["vote_real_fraction"] = n_real_votes / max(1, len(preds))

        print(
            f"[MODEL] multi-segment n={len(chunks)} vote_real={out['vote_real_fraction']:.2f} "
            f"raw_mean={raw_mean:.4f} raw_combined={raw_prob:.4f} pred={out['pred_label']}"
        )
        # region agent log
        _agent_debug_log(
            "H4",
            "model_predict_multisegment:exit",
            "segment_aggregate",
            {
                "n_segments": out["n_segments"],
                "vote_real_fraction": out["vote_real_fraction"],
                "segment_mean_raw_real": float(raw_mean),
                "raw_prob_combined": float(raw_prob),
                "pred_label": str(out["pred_label"]),
            },
        )
        # endregion
        return out
    except Exception as e:
        print(f"[MODEL] multi-segment error: {e}")
        out["error"] = str(e)
        return out


def _model_predict_singlepass(features44):
    """
    Fallback scorer when multi-segment path fails.
    Returns (raw_prob_real[0..1], pred_label, classes, real_idx).
    """
    features44 = np.asarray(features44, dtype=np.float32).flatten()
    # Prevent static outputs from invalid/empty feature vectors.
    if features44.size == 0 or not np.all(np.isfinite(features44)):
        raise RuntimeError("invalid feature vector for single-pass scoring")
    if np.linalg.norm(features44) < 1e-6 or float(np.std(features44)) < 1e-7:
        raise RuntimeError("degenerate feature vector for single-pass scoring")

    classes = list(getattr(model, "classes_", []))
    if not classes:
        raise RuntimeError("Model has no classes_")
    _infer_real_class_from_references(classes)
    real_idx = _get_real_class_index(classes)
    vin = _adapt_features_for_model(features44)
    pr = model.predict_proba([vin])[0]
    pred = model.predict([vin])[0]
    raw_prob_real = float(pr[real_idx])
    return raw_prob_real, pred, classes, real_idx


def _get_model_expected_features():
    """
    Returns expected feature count for model input, if discoverable.
    Handles sklearn pipelines and plain estimators.
    """
    n = getattr(model, "n_features_in_", None)
    if n is not None:
        return int(n)

    for attr in ("named_steps",):
        steps = getattr(model, attr, None)
        if isinstance(steps, dict):
            for step in steps.values():
                n = getattr(step, "n_features_in_", None)
                if n is not None:
                    return int(n)
    return None


def _adapt_features_for_model(features):
    """
    Keep inference robust if model expects 39/44/etc.
    """
    expected = _get_model_expected_features()
    if expected is None or expected == len(features):
        return features

    if len(features) > expected:
        print(f"[MODEL] Truncating features {len(features)} -> {expected}")
        return features[:expected]

    print(f"[MODEL] Padding features {len(features)} -> {expected}")
    return np.pad(features, (0, expected - len(features)))


def _get_real_class_index(classes):
    """
    Determine which class label means REAL.
    Training script uses: 1 = REAL, 0 = FAKE.
    """
    global REAL_CLASS_INDEX_OVERRIDE

    # If we successfully inferred from real reference voices, trust that first.
    if REAL_CLASS_INDEX_OVERRIDE is not None and REAL_CLASS_INDEX_OVERRIDE < len(classes):
        return REAL_CLASS_INDEX_OVERRIDE

    if 1 in classes:
        return classes.index(1)
    if "real" in classes:
        return classes.index("real")
    if "REAL" in classes:
        return classes.index("REAL")
    # Legacy fallback (older models may have inverted labels)
    if 0 in classes:
        return classes.index(0)
    return 0


def _infer_real_class_from_references(classes):
    """
    Calibrate real/fake class using known real reference voices.
    Useful when model labels differ from training convention.
    """
    global REAL_CLASS_INDEX_OVERRIDE
    if REAL_CLASS_INDEX_OVERRIDE is not None:
        return

    candidate_refs = []
    for fname in os.listdir(reference_folder):
        if fname.lower().endswith(".wav"):
            candidate_refs.append(os.path.join(reference_folder, fname))

    if not candidate_refs:
        return

    score_sum = np.zeros(len(classes), dtype=float)
    count = 0

    for ref_path in candidate_refs:
        try:
            feats = extract_features(ref_path)
            feats = _adapt_features_for_model(feats)
            probs = model.predict_proba([feats])[0]
            if len(probs) != len(classes):
                continue
            score_sum += probs
            count += 1
        except Exception as e:
            print(f"[MODEL] Reference calibration failed for {ref_path}: {e}")

    if count == 0:
        return

    avg_scores = score_sum / count
    REAL_CLASS_INDEX_OVERRIDE = int(np.argmax(avg_scores))
    print(
        f"[MODEL] Calibrated REAL class index from references: "
        f"{REAL_CLASS_INDEX_OVERRIDE} (avg={np.round(avg_scores, 4).tolist()})"
    )

# =========================
# COSINE SIMILARITY  (1 = identical, 0 = orthogonal, -1 = opposite)
# =========================
def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# =========================
# SPEAKER IDENTIFICATION
#
# Priority order:
#   1. name_hint  – politician detected from filename / URL / transcript
#   2. reference  – cosine similarity against reference WAV files
#   3. fallback   – "unknown"
# =========================
def identify_speaker(features, name_hint=None):
    """
    features  : np.ndarray from extract_features()
    name_hint : politician key pre-detected from text  (may be None)
    Returns   : (politician key string, best similarity float)
    """

    # Build reference list once for both hint and generic matching.
    ref_files = []
    for fname in sorted(os.listdir(reference_folder)):
        if fname.lower().endswith(".wav"):
            key = os.path.splitext(fname)[0].strip().lower()
            if key:
                ref_files.append((fname, key))

    # --- Priority 1: use hint, but still compute real similarity if possible ---
    if name_hint is not None:
        hint_key = str(name_hint).strip().lower()
        for fname, key in ref_files:
            if key == hint_key:
                ref_path = os.path.join(reference_folder, fname)
                try:
                    ref_feat = extract_features(ref_path)
                    sim = cosine_similarity(features, ref_feat)
                    print(f"[SPEAKER] Using name_hint: {hint_key} (sim={sim:.4f})")
                    return hint_key, float(sim)
                except Exception as e:
                    print(f"[SPEAKER] Hint similarity error for {fname}: {e}")
                    break

        print(f"[SPEAKER] Using name_hint: {hint_key} (no matching reference file)")
        return hint_key, 0.0

    # --- Priority 2: compare against reference WAV files ---
    best_name  = "unknown"
    best_score = -1.0          # cosine similarity – higher is better
    second_best_score = -1.0
    scores     = {}

    for fname, key in ref_files:
        ref_path = os.path.join(reference_folder, fname)
        try:
            ref_feat = extract_features(ref_path)
            sim      = cosine_similarity(features, ref_feat)
            scores[key] = round(sim, 4)
            print(f"[SPEAKER] {key} cosine sim = {sim:.4f}")
            if sim > best_score:
                second_best_score = best_score
                best_score = sim
                best_name  = key
            elif sim > second_best_score:
                second_best_score = sim
        except Exception as e:
            print(f"[SPEAKER] Error comparing {fname}: {e}")

    # Reject weak matches; allow lower floor so YouTube/generic filenames still match refs.
    if best_score < SPEAKER_REF_MIN_SIMILARITY:
        print(
            f"[SPEAKER] Best score {best_score:.4f} below min {SPEAKER_REF_MIN_SIMILARITY} -> unknown"
        )
        best_name = "unknown"
    elif second_best_score >= SPEAKER_REF_SECOND_AMBIGUOUS and (
        best_score - second_best_score
    ) < SPEAKER_REF_AMBIGUITY_MARGIN:
        # Cosine on global MFCC often ties across refs; still pick argmax so
        # calibration + UI are not blocked by "unknown" (debug session db268d).
        print(
            f"[SPEAKER] Ambiguous cosine (keep argmax): top1={best_name}={best_score:.4f} "
            f"top2={second_best_score:.4f} margin={(best_score-second_best_score):.4f}"
        )

    print(f"[SPEAKER] Final: {best_name}  (scores={scores})")
    return best_name, float(best_score)


def calibrate_confidence_for_reference_match(
    confidence,
    raw_prob_real,
    person,
    speaker_sim,
    pred_is_real,
    source_label=None,
    name_hint=None,
    vote_real_fraction=None,
    segment_mean_raw=None,
    n_segments=None,
):
    """
    Help real YouTube clips when RF predicts REAL but probability is <90%.

    Optional "hint + segments" path: if the aggregate class is FAKE but the filename
    hints a known politician, that speaker matches the hint, most windows lean REAL,
    and cosine to that ref is decent, still apply the reference blend (conservative
    gates) — reduces false FAKE on real Najib/Mahathir YouTube uploads.
    """
    vote = float(vote_real_fraction or 0.0)
    segm = float(segment_mean_raw or 0.0)
    nseg = int(n_segments or 0)
    hint = (name_hint or "").strip().lower()
    pers = (person or "").strip().lower()
    src = str(source_label or "").strip().lower()
    is_upload_source = src.startswith("upload:")

    # Uploads are most vulnerable to cloned voices with high speaker similarity.
    # Keep this reference-based REAL boost for URL/live only (can be re-enabled via env).
    allow_upload_ref_boost = os.environ.get(
        "MV_UPLOAD_ALLOW_REFERENCE_REAL_BOOST", "0"
    ).strip().lower() not in ("0", "false", "no", "off")
    if is_upload_source and not allow_upload_ref_boost:
        return confidence

    cross = (
        not pred_is_real
        and hint in KNOWN_POLITICIAN_KEYS
        and hint == pers
        and vote >= 0.38
        and segm >= 0.24
        and nseg >= 2
        and speaker_sim >= 0.35
    )

    if not pred_is_real and not cross:
        return confidence

    min_sim = 0.36 if cross else 0.48
    min_raw = 0.28 if cross else 0.40
    if speaker_sim < min_sim:
        return confidence
    if raw_prob_real < min_raw:
        return confidence
    # Borderline gates (slightly looser when cross-class YouTube path)
    if cross:
        if raw_prob_real < 0.38 and speaker_sim < 0.46:
            return confidence
    else:
        if raw_prob_real < 0.48 and speaker_sim < 0.58:
            return confidence
    if confidence >= THRESHOLD_REAL:
        return confidence

    if cross:
        alt = 14.0 + 102.0 * float(raw_prob_real) + 62.0 * float(speaker_sim)
    else:
        alt = 12.0 + 78.0 * float(raw_prob_real) + 44.0 * float(speaker_sim)

    candidates = [confidence, 100.0 * float(raw_prob_real), alt]
    if cross and vote >= 0.42:
        # Push borderline REAL clips over the Najib/Mahathir bar when ref + votes agree
        floor_c = 56.0 + 40.0 * vote + 40.0 * float(speaker_sim)
        candidates.append(floor_c)
    new_conf = float(min(99.0, max(candidates)))
    if new_conf > confidence + 0.01:
        tag = "hint+segments+ref" if cross else "reference-aware"
        print(
            f"[MODEL] {tag} confidence: base={confidence:.2f} "
            f"raw_real={raw_prob_real:.4f} sim={speaker_sim:.4f} -> {new_conf:.2f}"
        )
    return round(new_conf, 2)


# =========================
# SIGNATURE-BASED DETECTION  (filename artifacts)
# =========================
AI_MARKERS = [
    "tts", "generated", "synthesized", "deepfake",
    "artificial", "elevenlabs", "murf", "fakeyou", "voiceclone",
]

def check_signature(path):
    name = os.path.basename(path).lower()
    for marker in AI_MARKERS:
        if marker in name:
            print(f"[SIGNATURE] AI marker '{marker}' in filename")
            return True
    return False


def _live_webrtc_vad_speech_fraction(y, sr=16000):
    """
    Fraction of ~30ms frames WebRTC VAD classifies as speech.
    Steady fan/hiss often passes RMS gates but fails VAD.
    Returns None if webrtcvad is not installed.
    """
    try:
        import webrtcvad  # type: ignore
    except ImportError:
        return None
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < 480:
        return 0.0
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    vad = webrtcvad.Vad(3)
    frame_len = 480  # 30ms @ 16kHz
    hop = 240  # 15ms hop
    voiced = 0
    nframes = 0
    for start in range(0, pcm.size - frame_len + 1, hop):
        frame = pcm[start : start + frame_len]
        if vad.is_speech(frame.tobytes(), int(sr)):
            voiced += 1
        nframes += 1
    return float(voiced) / float(max(1, nframes))


def _live_fallback_steady_noise_screen(y, sr=16000):
    """
    When webrtcvad is missing: block steady broadband noise (no amplitude modulation).
    May rarely affect monotone speech — set MV_LIVE_USE_STEADY_NOISE_SCREEN=0 to disable.
    """
    if os.environ.get("MV_LIVE_USE_STEADY_NOISE_SCREEN", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False, None
    rms_f = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if rms_f.size < 4:
        return False, None
    mu = float(np.mean(rms_f)) + 1e-10
    cv = float(np.std(rms_f) / mu)
    flat = librosa.feature.spectral_flatness(y=y)[0]
    flat_m = float(np.mean(flat))
    # Tunables via env
    try:
        cv_max = float(os.environ.get("MV_LIVE_STEADY_CV_MAX", "0.22").strip() or "0.22")
    except Exception:
        cv_max = 0.22
    try:
        flat_min = float(os.environ.get("MV_LIVE_STEADY_FLAT_MIN", "0.22").strip() or "0.22")
    except Exception:
        flat_min = 0.22
    if cv < cv_max and flat_m > flat_min:
        return True, {"rms_cv": cv, "spectral_flatness_mean": flat_m}
    return False, {"rms_cv": cv, "spectral_flatness_mean": flat_m}


def _live_speech_suitability(wav_path, transcript_empty=False):
    """
    The RF is trained on speech; silence / room-noise is out-of-distribution and
    often scores ~90%+ REAL (verified: digital silence -> 94% REAL, db268d).
    Returns (ok: bool, metrics: dict, reason: str | None).

    Browser Web Speech often hallucinates words on fan noise — so we no longer
    use transcript_empty to *loosen* the gate. With MV_LIVE_ALWAYS_STRICT=1 (default),
    live clips always use strong RMS / voiced-frame floors. Set MV_LIVE_ALWAYS_STRICT=0
    to use env/base thresholds only.
    """
    if not LIBROSA_OK:
        return True, {"librosa": False}, None
    try:
        y, sr = _load_audio_mono_16k(wav_path)
    except Exception as e:
        return True, {"load_error": str(e)[:80]}, None
    y = np.asarray(y, dtype=np.float32)
    duration = float(len(y) / float(sr))
    peak = float(np.max(np.abs(y)))
    g_rms = float(np.sqrt(np.mean(np.square(y))))
    rms_f = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_p50 = float(np.percentile(rms_f, 50))
    rms_p90 = float(np.percentile(rms_f, 90))
    try:
        p90_min = float(os.environ.get("MV_LIVE_RMS_P90_MIN", "0.014").strip() or "0.014")
    except Exception:
        p90_min = 0.014
    try:
        frac_min = float(os.environ.get("MV_LIVE_FRAC_VOICED_MIN", "0.17").strip() or "0.17")
    except Exception:
        frac_min = 0.17
    frame_thr = 0.021
    always_strict = os.environ.get("MV_LIVE_ALWAYS_STRICT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if always_strict:
        # Strong floors for every live take — noise + fake STT transcript cannot bypass.
        p90_min = max(p90_min, 0.022)
        frac_min = max(frac_min, 0.26)
        frame_thr = max(frame_thr, 0.032)
    frac_voiced = float(np.mean(rms_f > frame_thr))
    metrics = {
        "duration_sec": round(duration, 3),
        "global_rms": g_rms,
        "peak": peak,
        "rms_p50": rms_p50,
        "rms_p90": rms_p90,
        "frame_thr_used": frame_thr,
        "frac_frames_above_thr": round(frac_voiced, 4),
        "transcript_empty": transcript_empty,
        "always_strict": bool(always_strict),
    }
    if duration < 0.35:
        return False, metrics, "clip_too_short"
    if peak < 2e-4 and g_rms < 5e-5:
        return False, metrics, "silence_or_disconnected"
    if duration >= 0.5 and rms_p90 < p90_min and frac_voiced < frac_min:
        return False, metrics, "no_clear_speech"

    vad_f = _live_webrtc_vad_speech_fraction(y, sr)
    metrics["vad_speech_fraction"] = None if vad_f is None else round(float(vad_f), 4)
    try:
        vad_min = float(os.environ.get("MV_LIVE_VAD_MIN", "0.10").strip() or "0.10")
    except Exception:
        vad_min = 0.10
    # Loud HVAC/hiss can satisfy RMS; VAD checks speech-like voicing.
    if vad_f is not None and duration >= 0.4 and vad_f < vad_min:
        return False, metrics, "no_voice_activity_vad"
    if vad_f is None:
        steady, extra = _live_fallback_steady_noise_screen(y, sr)
        if isinstance(extra, dict):
            metrics["steady_noise_screen"] = extra
        if steady and duration >= 0.4:
            return False, metrics, "no_speech_steady_noise"

    return True, metrics, None


# =========================
# MAIN PIPELINE
# =========================
def process_audio(
    wav_path,
    source_label,
    name_hint=None,
    audio_start_sec=0.0,
    live_transcript=None,
):
    start = time.time()
    print(f"\n[PIPELINE] ===== START ====="
          f"\n[PIPELINE] file={wav_path}  hint={name_hint}  source={source_label}"
          f"\n[PIPELINE] audio_start_sec={audio_start_sec}")

    # Step 1 – signature check on source label / original filename
    if check_signature(source_label) or check_signature(wav_path):
        person = name_hint or "unknown"
        return {
            "confidence": 20.0,
            "person":     person,
            "time":       round(time.time() - start, 2),
            "source":     source_label,
            "result":     "FAKE",
        }

    # Step 1b – live only: do not trust deepfake score when there is no usable speech
    if (source_label or "").strip() == "Live Recording":
        t_empty = not (live_transcript or "").strip()
        ok, sp_metrics, sp_reason = _live_speech_suitability(
            wav_path, transcript_empty=t_empty
        )
        # region agent log
        _agent_debug_log(
            "H1",
            "process_audio:live_speech_gate",
            "suitability",
            {"ok": ok, "reason": sp_reason, "metrics": sp_metrics},
            run_id="post-fix",
        )
        # endregion
        if not ok:
            print(
                f"[LIVE-GATE] blocked reason={sp_reason} metrics={sp_metrics}"
            )
            msg = {
                "clip_too_short": "Recording is too short. Please record at least a few seconds of clear speech.",
                "silence_or_disconnected": "No usable audio (silence or very low level). Check the microphone, then try again.",
                "no_clear_speech": "No clear speech detected (only silence or background noise). Speak close to the mic, then try again.",
                "no_voice_activity_vad": "No speech-like voice activity detected (quiet room / fan / hiss only). Speak clearly toward the microphone.",
                "no_speech_steady_noise": "Audio looks like steady noise only (no speech bursts). Speak clearly, or install optional pip package webrtcvad for better filtering.",
            }.get(sp_reason or "", "No clear speech detected. Please try again.")
            return {
                "confidence": 0.0,
                "person":     "unknown",
                "time":       round(time.time() - start, 2),
                "source":     source_label,
                "result":     "FAKE",
                "speaker_similarity": 0.0,
                "model_says_real":    False,
                "segment_count":      0,
                "vote_real_fraction": 0.0,
                "threshold_used":     int(THRESHOLD_REAL),
                "insufficient_speech": True,
                "speech_gate_reason":  sp_reason,
                "speech_gate_message": msg,
                "speech_metrics":      sp_metrics,
            }

    # Step 2–3 – per-window RF scores, then aggregate (not mean-MFCC + one score)
    raw_prob_real = 0.5
    pred_label = None
    classes = []
    real_idx = 0
    features = np.zeros(44)
    vote_real = 0.0
    n_seg = 0
    seg_mean = 0.0
    try:
        mp = model_predict_multisegment(wav_path, start_offset_sec=float(audio_start_sec or 0.0))
        features = mp["features_for_speaker"]
        raw_prob_real = float(mp["raw_prob_real"])
        pred_label = mp["pred_label"]
        classes = mp["classes"]
        real_idx = int(mp["real_idx"])
        vote_real = float(mp["vote_real_fraction"])
        n_seg = int(mp["n_segments"])
        seg_mean = float(mp["segment_mean_raw_real"])

        # If most windows say REAL but combined prob is borderline, nudge upward
        if vote_real >= 0.52 and seg_mean >= 0.38 and raw_prob_real < seg_mean:
            raw_prob_real = float(min(0.97, max(raw_prob_real, seg_mean + 0.04)))

        prob_real = max(0.0, min(100.0, raw_prob_real * 100.0))
        print(
            f"[MODEL] classes={classes} real_idx={real_idx} pred={pred_label} "
            f"raw_real={raw_prob_real:.4f} prob_real={prob_real:.2f}"
        )
        # If multi-segment returned neutral/default output, run single-pass model fallback
        # instead of returning 50% for every file.
        if (
            (not classes)
            or pred_label is None
            or ("error" in mp)
            or (n_seg == 0 and abs(raw_prob_real - 0.5) < 1e-9)
        ):
            print(
                "[MODEL] multi-segment fallback -> single-pass predict_proba "
                f"(reason error={mp.get('error') if isinstance(mp, dict) else None})"
            )
            if np.linalg.norm(features) < 1e-9:
                features = extract_features_for_predict(
                    wav_path, start_offset_sec=float(audio_start_sec or 0.0)
                )
            raw_prob_real, pred_label, classes, real_idx = _model_predict_singlepass(features)
            prob_real = max(0.0, min(100.0, raw_prob_real * 100.0))
            vote_real = raw_prob_real
            seg_mean = raw_prob_real
            n_seg = max(1, n_seg)
            print(
                f"[MODEL] single-pass classes={classes} real_idx={real_idx} pred={pred_label} "
                f"raw_real={raw_prob_real:.4f} prob_real={prob_real:.2f}"
            )
    except Exception as e:
        print(f"[MODEL] Error: {e}")
        prob_real = 50.0
        raw_prob_real = 0.5
        try:
            features = extract_features_for_predict(
                wav_path, start_offset_sec=float(audio_start_sec or 0.0)
            )
            try:
                raw_prob_real, pred_label, classes, real_idx = _model_predict_singlepass(features)
                prob_real = max(0.0, min(100.0, raw_prob_real * 100.0))
                vote_real = raw_prob_real
                seg_mean = raw_prob_real
                n_seg = max(1, n_seg)
                print(
                    f"[MODEL] exception fallback single-pass raw_real={raw_prob_real:.4f} "
                    f"prob_real={prob_real:.2f}"
                )
            except Exception as e2:
                print(f"[MODEL] fallback single-pass error: {e2}")
        except Exception:
            features = np.zeros(44)

    confidence = round(prob_real, 2)

    pred_is_real = False
    if pred_label is not None and classes and len(classes) > real_idx:
        real_class = classes[real_idx]
        pred_is_real = (pred_label == real_class) or (str(pred_label) == str(real_class))
        # Segment majority: helps long YouTube where mean-argmax class disagrees with most trees
        if not pred_is_real:
            if vote_real >= 0.55 and seg_mean >= 0.40 and n_seg >= 4:
                pred_is_real = True
                print(
                    f"[MODEL] pred_is_real=True from segment majority "
                    f"(votes={vote_real:.2f} mean_raw={seg_mean:.4f} n={n_seg})"
                )
            elif (
                (name_hint or "").strip().lower() in ("anwar", "najib", "mahathir")
                and vote_real >= 0.40
                and seg_mean >= 0.26
                and n_seg >= 2
            ):
                pred_is_real = True
                print(
                    f"[MODEL] pred_is_real=True (hint + soft segment vote "
                    f"votes={vote_real:.2f} mean_raw={seg_mean:.4f} n={n_seg})"
                )

    is_upload_source = str(source_label or "").strip().lower().startswith("upload:")

    # Step 4 – speaker identification (uses hint first, then reference comparison)
    person, speaker_similarity = identify_speaker(features, name_hint=name_hint)

    # Step 4b – reference blend (and optional hint+segment path for borderline YouTube REAL)
    confidence_before_calibration = float(confidence)
    confidence = calibrate_confidence_for_reference_match(
        confidence,
        raw_prob_real,
        person,
        speaker_similarity,
        pred_is_real,
        source_label=source_label,
        name_hint=name_hint,
        vote_real_fraction=vote_real,
        segment_mean_raw=seg_mean,
        n_segments=n_seg,
    )
    # region agent log
    _agent_debug_log(
        "H_CONF",
        "process_audio:after_calibration",
        "confidence_decision_trace",
        {
            "source_label": source_label,
            "name_hint": name_hint,
            "person": person,
            "pred_is_real": bool(pred_is_real),
            "raw_prob_real": float(raw_prob_real),
            "vote_real_fraction": float(vote_real),
            "segment_mean_raw": float(seg_mean),
            "speaker_similarity": float(speaker_similarity),
            "confidence_before_calibration": float(confidence_before_calibration),
            "confidence_after_calibration": float(confidence),
            "threshold_real": int(THRESHOLD_REAL),
        },
        run_id="pre-fix",
    )
    # endregion

    # Guardrail: if RF aggregate leans FAKE strongly, never allow post-processing
    # to keep/raise a REAL verdict (especially important for cloned uploads).
    if (
        is_upload_source
        and not bool(pred_is_real)
        and (float(raw_prob_real) <= 0.62 or float(vote_real) <= 0.45)
    ):
        old_conf = float(confidence)
        confidence = float(min(confidence, THRESHOLD_REAL - 0.1))
        print(
            f"[UPLOAD-FAKE-GUARD] enforce FAKE: pred_is_real={pred_is_real} "
            f"raw={raw_prob_real:.4f} vote={vote_real:.2f} conf {old_conf:.2f}->{confidence:.2f}"
        )

    # Step 4c – optional stricter upload policy (off by default; matches pre-regression UX).
    # Set MV_UPLOAD_STRICT=1 to require extra RF agreement before REAL on file uploads.
    upload_strict = os.environ.get("MV_UPLOAD_STRICT", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if is_upload_source and upload_strict:
        hinted = bool((name_hint or "").strip())
        min_raw = 0.94 if hinted else 0.90
        min_vote = 0.82 if hinted else 0.72
        if not (
            bool(pred_is_real)
            and float(raw_prob_real) >= min_raw
            and float(vote_real) >= min_vote
        ):
            old_conf = float(confidence)
            confidence = float(min(confidence, THRESHOLD_REAL - 0.1))
            print(
                f"[UPLOAD-STRICT] force FAKE gate: pred_is_real={pred_is_real} "
                f"raw={raw_prob_real:.4f} vote={vote_real:.2f} hinted={hinted} "
                f"conf {old_conf:.2f}->{confidence:.2f}"
            )
            # region agent log
            _agent_debug_log(
                "H_UPLOAD_STRICT",
                "process_audio:upload_gate",
                "upload_strict_gate_applied",
                {
                    "pred_is_real": bool(pred_is_real),
                    "raw_prob_real": float(raw_prob_real),
                    "vote_real_fraction": float(vote_real),
                    "hinted": hinted,
                    "min_raw": float(min_raw),
                    "min_vote": float(min_vote),
                    "old_confidence": float(old_conf),
                    "new_confidence": float(confidence),
                },
                run_id="post-fix",
            )
            # endregion

    # Final binary verdict: strict global threshold for all sources.
    threshold_used = float(THRESHOLD_REAL)
    result = "REAL" if confidence >= threshold_used else "FAKE"

    elapsed = round(time.time() - start, 2)
    print(
        f"[PIPELINE] Done: {result} {confidence}%  speaker={person}  "
        f"threshold={threshold_used}%  pred_is_real={pred_is_real}  time={elapsed}s\n"
    )

    return {
        "confidence": confidence,
        "person":     person,
        "time":       elapsed,
        "source":     source_label,
        "result":     result,
        "speaker_similarity": round(float(speaker_similarity), 4),
        "model_says_real":    bool(pred_is_real),
        "segment_count":      int(n_seg),
        "vote_real_fraction": round(float(vote_real), 4),
        "threshold_used":     float(round(threshold_used, 2)),
    }

def safe_remove(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

# =========================
# ROUTES
# =========================
@app.route("/health", methods=["GET"])
def health():
    refs = os.listdir(reference_folder) if os.path.exists(reference_folder) else []
    classes_attr = getattr(model, "classes_", None)
    if classes_attr is None:
        model_classes = []
    else:
        model_classes = [str(x) for x in list(classes_attr)]

    _seg_def = _infer_segment_defaults()
    payload = {
        "status":           "OK",
        "live_pipeline_version": "7-render-oom-mitigations",
        "model":            model_status,
        "librosa":          LIBROSA_OK,
        "pydub":            PYDUB_OK,
        "ffmpeg_in_path":   _ffmpeg_available(),
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
        "infer_low_memory_mode": _infer_low_memory_mode(),
        "infer_max_audio_sec":   _infer_max_audio_seconds(),
        "infer_segment_defaults": {
            "segment_sec": _seg_def[0],
            "hop_sec": _seg_def[1],
            "max_segments_default": _seg_def[2],
        },
        "model_classes":    model_classes,
        "model_features":   _get_model_expected_features(),
        "reference_voices": refs,
        "thresholds":       {
            "real": THRESHOLD_REAL,
            "fake": THRESHOLD_FAKE,
            "policy": "binary: >=real is REAL, else FAKE",
            "known_speaker_min_similarity": KNOWN_SPEAKER_MIN_SIMILARITY,
            "speaker_ref_min_similarity": SPEAKER_REF_MIN_SIMILARITY,
            "speaker_ref_second_ambiguous": SPEAKER_REF_SECOND_AMBIGUOUS,
            "speaker_ref_ambiguity_margin": SPEAKER_REF_AMBIGUITY_MARGIN,
        },
    }

    # Optional dataset / model / reference scan (same data as python diagnose.py).
    # GET /health?diagnostics=1          — fast scan + strict_issues list
    # GET /health?diagnostics=full       — also refs_cosine (librosa; slower)
    # Set ENABLE_HEALTH_DIAGNOSTICS=0 to disable even when query params are sent.
    diag_q = (request.args.get("diagnostics") or "").strip().lower()
    env_diag = os.environ.get("ENABLE_HEALTH_DIAGNOSTICS", "1").strip().lower()
    diag_disabled = env_diag in ("0", "false", "no", "off")

    if diag_q in ("1", "true", "yes", "full"):
        if diag_disabled:
            payload["diagnostics"] = {
                "error": "Health diagnostics disabled (set ENABLE_HEALTH_DIAGNOSTICS=1 to allow).",
            }
        else:
            try:
                from diagnose import build_diagnostics_payload

                payload["diagnostics"] = build_diagnostics_payload(
                    include_refs_cosine=(diag_q == "full"),
                    include_strict_issues=True,
                )
            except Exception as e:
                payload["diagnostics"] = {"error": str(e)}

    return jsonify(payload)

@app.route("/routes", methods=["GET"])
def show_routes():
    return jsonify([
        {"path": str(r), "methods": list(r.methods)}
        for r in app.url_map.iter_rules()
    ])

# ------------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------------
@app.route("/predict-file", methods=["POST", "OPTIONS"])
def predict_file():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    try:
        print(f"[UPLOAD] request.files keys: {list(request.files.keys())}")
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_file:entry",
            "entered_predict_file",
            {
                "content_type": request.content_type,
                "files_keys": list(request.files.keys()),
            },
            run_id="pre-fix",
        )
        # endregion

        if "file" not in request.files:
            return jsonify({"error": "No 'file' key in FormData"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # --- Size check ---
        file.seek(0, 2); size = file.tell(); file.seek(0)
        print(f"[UPLOAD] {file.filename}  {size/1024:.1f}KB  {file.content_type}")
        if size > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large ({size/1024/1024:.1f} MB, max 50 MB)"}), 400

        # --- Extension check ---
        ext = os.path.splitext(file.filename)[1].lower()
        allowed = {".mp3", ".wav", ".ogg", ".m4a", ".webm"}
        if ext not in allowed:
            return jsonify({"error": f"Unsupported type '{ext}'. Allowed: MP3, WAV, OGG, M4A, WEBM"}), 400

        # ------------------------------------------------------------------
        # Filename hint: used for speaker card alignment + calibration paths
        # (cosine-only matching often picks the wrong ref when MFCC space overlaps).
        # Disable with MV_UPLOAD_USE_FILENAME_HINT=0 if filenames are unreliable.
        # ------------------------------------------------------------------
        detected_hint = detect_politician_from_text(file.filename)
        _hint_env = os.environ.get("MV_UPLOAD_USE_FILENAME_HINT", "1").strip().lower()
        use_filename_hint = _hint_env not in ("0", "false", "no", "off")
        name_hint = detected_hint if use_filename_hint else None
        print(
            f"[UPLOAD] detected_hint={detected_hint} "
            f"use_filename_hint={use_filename_hint} applied_hint={name_hint}"
        )

        uid      = str(uuid.uuid4())[:8]
        in_path  = os.path.join(temp_folder, f"in_{uid}{ext}")
        wav_path = os.path.join(temp_folder, f"proc_{uid}.wav")

        file.save(in_path)

        if not convert_to_wav(in_path, wav_path):
            safe_remove(in_path)
            return jsonify(
                {
                    "error": (
                        "Audio conversion failed (MP3/M4A/WebM need ffmpeg on the server). "
                        "On Render: deploy with Docker using the repo Dockerfile, or install ffmpeg. "
                        f"ffmpeg_in_path={_ffmpeg_available()}"
                    )
                }
            ), 500

        source_label = f"Upload: {file.filename[:50]}"
        data = process_audio(wav_path, source_label, name_hint=name_hint)
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_file:before_return",
            "predict_file_success",
            {
                "result": data.get("result"),
                "confidence": data.get("confidence"),
                "person": data.get("person"),
            },
            run_id="pre-fix",
        )
        # endregion

        safe_remove(in_path, wav_path)
        return jsonify(data)

    except Exception as e:
        import traceback
        print(f"[ERROR] predict_file:\n{traceback.format_exc()}")
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_file:except",
            "predict_file_exception",
            {"error": str(e)},
            run_id="pre-fix",
        )
        # endregion
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# URL / SOCIAL MEDIA
# ------------------------------------------------------------------
@app.route("/predict-url", methods=["POST", "OPTIONS"])
def predict_url():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    try:
        body = request.get_json(force=True, silent=True) or {}
        url  = body.get("url", "").strip()
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_url:entry",
            "entered_predict_url",
            {
                "content_type": request.content_type,
                "has_url": bool(url),
            },
            run_id="pre-fix",
        )
        # endregion
        if not url:
            return jsonify({"error": "Missing 'url' in JSON body"}), 400

        print(f"[URL] {url}")

        yt_audio_start_sec = _parse_youtube_start_seconds(url)
        if yt_audio_start_sec > 0:
            print(
                f"[URL] Scoring uses speech from {yt_audio_start_sec:.2f}s "
                f"(from ?t= / &start=; avoids intro music before the speech)"
            )

        # ------------------------------------------------------------------
        # KEY FIX: detect politician from the URL string itself
        # e.g. youtube.com/watch?v=…&title=Anwar → hint="anwar"
        # Also check the optional "videoTitle" field sent from the frontend
        # ------------------------------------------------------------------
        hint_text = url + " " + body.get("videoTitle", "")
        name_hint = detect_politician_from_text(hint_text)
        print(f"[URL] name_hint from url+title: {name_hint}")

        uid        = str(uuid.uuid4())[:8]
        wav_path   = os.path.join(temp_folder, f"url_{uid}.wav")
        downloaded = None

        social = [
            "youtube.com", "youtu.be", "tiktok.com",
            "instagram.com", "twitter.com", "x.com", "facebook.com",
        ]

        if any(s in url.lower() for s in social):
            if not YTDLP_OK:
                return jsonify({"error": "yt-dlp not installed: pip install yt-dlp"}), 500

            video_title = ""
            uploader = ""
            description = ""
            # ------------------------------------------------------------------
            # Title + politician hint: yt-dlp metadata, then oEmbed fallback
            # ------------------------------------------------------------------
            meta_opts = {
                "quiet": True,
                "skip_download": True,
                "noplaylist": True,
                "extractor_args": _yt_dlp_youtube_extractor_args(),
            }
            try:
                with yt_dlp.YoutubeDL(meta_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_title = (info.get("title") or "").strip()
                    uploader = (info.get("uploader") or "").strip()
                    description = ((info.get("description") or "")[:300]).strip()
                    print(f"[URL] yt-dlp title: {video_title!r} uploader: {uploader!r}")
            except Exception as e:
                print(f"[URL] yt-dlp metadata extraction failed: {e}")

            if not video_title:
                video_title = _youtube_oembed_title(url)
                if video_title:
                    print(f"[URL] oEmbed title: {video_title!r}")

            meta_text = f"{video_title} {uploader} {description}"
            if name_hint is None:
                name_hint = detect_politician_from_text(meta_text)
                print(f"[URL] name_hint from video meta: {name_hint}")

            # region agent log
            _agent_debug_log(
                "H2_H3",
                "predict_url:after_meta",
                "youtube_meta_and_hint",
                {
                    "url_has_start_param": ("t=" in url or "start=" in url.lower()),
                    "parsed_yt_start_sec": yt_audio_start_sec,
                    "video_title_len": len(video_title or ""),
                    "video_title_snip": (video_title or "")[:100],
                    "name_hint": name_hint,
                },
            )
            # endregion

            tmpl = os.path.join(temp_folder, f"yt_{uid}.%(ext)s")
            dl_opts = {
                "format": (
                    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/"
                    "ba/bestaudio/best/ba/best"
                ),
                "outtmpl": tmpl,
                "quiet": True,
                "noplaylist": True,
                "retries": 5,
                "fragment_retries": 5,
                "extractor_args": _yt_dlp_youtube_extractor_args(),
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])

            for f in os.listdir(temp_folder):
                if f.startswith(f"yt_{uid}"):
                    downloaded = os.path.join(temp_folder, f)
                    break

            if downloaded and os.path.isfile(downloaded):
                sz = os.path.getsize(downloaded)
                print(f"[URL] Downloaded bytes: {sz}")
                # region agent log
                _agent_debug_log(
                    "H1",
                    "predict_url:after_download",
                    "yt_download",
                    {
                        "size_bytes": sz,
                        "ext": os.path.splitext(downloaded)[1].lower(),
                        "basename": os.path.basename(downloaded),
                    },
                )
                # endregion
                if sz < 8000:
                    safe_remove(downloaded)
                    return jsonify(
                        {
                            "error": (
                                "YouTube download was too small (broken link, private video, "
                                "or wrong URL). Copy the full watch URL from the browser bar "
                                "(watch for typos: I vs l in the video id)."
                            )
                        }
                    ), 400

        else:
            if not REQUESTS_OK:
                return jsonify({"error": "requests not installed"}), 500

            r = req_lib.get(
                url, timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
                stream=True,
            )
            if r.status_code != 200:
                return jsonify({"error": f"HTTP {r.status_code}"}), 400

            url_ext = url.lower().split("?")[0].rsplit(".", 1)[-1]
            if url_ext not in ["mp3", "wav", "ogg", "m4a", "webm"]:
                url_ext = "mp3"
            downloaded = os.path.join(temp_folder, f"dl_{uid}.{url_ext}")
            with open(downloaded, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        if not downloaded or not os.path.exists(downloaded):
            return jsonify({"error": "Download failed"}), 500

        if not convert_to_wav(downloaded, wav_path):
            safe_remove(downloaded)
            return jsonify({"error": "Conversion failed"}), 500

        dur = _wav_duration_seconds(wav_path)
        # region agent log
        _agent_debug_log(
            "H1",
            "predict_url:after_convert",
            "wav_duration",
            {"dur_seconds": dur, "wav_basename": os.path.basename(wav_path)},
        )
        # endregion
        if dur is not None and dur < 1.5:
            safe_remove(downloaded, wav_path)
            return jsonify(
                {
                    "error": (
                        f"Converted audio is only {dur:.1f}s — download likely failed or URL is wrong. "
                        "Use the full youtube.com/watch?v=… link and check the video id for typos."
                    )
                }
            ), 400

        source_label = f"URL: {url[:60]}{'...' if len(url) > 60 else ''}"
        data = process_audio(
            wav_path, source_label, name_hint=name_hint, audio_start_sec=yt_audio_start_sec
        )
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_url:before_return",
            "predict_url_success",
            {
                "result": data.get("result"),
                "confidence": data.get("confidence"),
                "person": data.get("person"),
            },
            run_id="pre-fix",
        )
        # endregion

        # region agent log
        _agent_debug_log(
            "H5",
            "predict_url:after_process_audio",
            "api_result",
            {
                "parsed_yt_start_sec": yt_audio_start_sec,
                "confidence": data.get("confidence"),
                "result": data.get("result"),
                "model_says_real": data.get("model_says_real"),
                "person": data.get("person"),
                "vote_real_fraction": data.get("vote_real_fraction"),
                "segment_count": data.get("segment_count"),
                "speaker_similarity": data.get("speaker_similarity"),
                "name_hint_passed": name_hint,
            },
        )
        # endregion

        safe_remove(downloaded, wav_path)
        return jsonify(data)

    except Exception as e:
        import traceback
        print(f"[ERROR] predict_url:\n{traceback.format_exc()}")
        # region agent log
        _agent_debug_log(
            "H_REQ",
            "predict_url:except",
            "predict_url_exception",
            {"error": str(e)},
            run_id="pre-fix",
        )
        # endregion
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# LIVE RECORDING
# ------------------------------------------------------------------
@app.route("/predict-live", methods=["POST", "OPTIONS"])
def predict_live():
    """
    Dedicated endpoint for live recordings.
    Accepts multipart: file=<audio blob>  transcript=<spoken text>
    """
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    try:
        if "file" not in request.files:
            return jsonify({"error": "No 'file' key in FormData"}), 400

        file       = request.files["file"]
        transcript = request.form.get("transcript", "").strip()

        print(f"[LIVE] filename={file.filename}  transcript='{transcript[:80]}'")

        # ------------------------------------------------------------------
        # KEY FIX: detect politician from the spoken transcript
        # e.g. user says "Anwar Ibrahim speech" → hint="anwar"
        # ------------------------------------------------------------------
        name_hint = detect_politician_from_text(transcript) if transcript else None
        print(f"[LIVE] name_hint from transcript: {name_hint}")

        file.seek(0, 2); size = file.tell(); file.seek(0)
        if size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400

        ext      = os.path.splitext(file.filename)[1].lower() or ".webm"
        uid      = str(uuid.uuid4())[:8]
        in_path  = os.path.join(temp_folder, f"live_{uid}{ext}")
        wav_path = os.path.join(temp_folder, f"live_{uid}.wav")

        file.save(in_path)

        if not convert_to_wav(in_path, wav_path):
            safe_remove(in_path)
            return jsonify({"error": "Audio conversion failed"}), 500

        t_live = time.time()
        # When the user does not speak, Web Speech often returns "" — the RF still
        # scores ~90% REAL on room noise. Skip the model unless disabled via env.
        skip_no_tr = os.environ.get("MV_LIVE_SKIP_MODEL_IF_NO_TRANSCRIPT", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        if skip_no_tr and not transcript:
            print("[LIVE] Empty transcript -> 0% FAKE (no model). MV_LIVE_SKIP_MODEL_IF_NO_TRANSCRIPT=0 to allow.")
            safe_remove(in_path, wav_path)
            return jsonify(
                {
                    "confidence": 0.0,
                    "person":     "unknown",
                    "time":       round(time.time() - t_live, 2),
                    "source":     "Live Recording",
                    "result":     "FAKE",
                    "speaker_similarity": 0.0,
                    "model_says_real":    False,
                    "segment_count":      0,
                    "vote_real_fraction": 0.0,
                    "threshold_used":     int(THRESHOLD_REAL),
                    "insufficient_speech": True,
                    "speech_gate_reason":  "no_speech_recognized",
                    "speech_gate_message": (
                        "No speech was recognized. Speak clearly near the microphone, then try again. "
                        "If you are speaking in another language, your server may need "
                        "MV_LIVE_SKIP_MODEL_IF_NO_TRANSCRIPT=0 so analysis can use audio only."
                    ),
                }
            )

        data = process_audio(
            wav_path,
            "Live Recording",
            name_hint=name_hint,
            live_transcript=transcript,
        )

        safe_remove(in_path, wav_path)
        return jsonify(data)

    except Exception as e:
        import traceback
        print(f"[ERROR] predict_live:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# =========================
# START
# =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MyVoiceGuard Backend")
    print("  http://127.0.0.1:5000")
    print("  http://127.0.0.1:5000/health")
    print(f"  Model : {model_status}")
    print(f"  librosa: {'YES' if LIBROSA_OK else 'NO - pip install librosa'}")
    print(f"  pydub  : {'YES' if PYDUB_OK   else 'NO - pip install pydub'}")
    print(f"  refs   : {os.listdir(reference_folder)}")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
