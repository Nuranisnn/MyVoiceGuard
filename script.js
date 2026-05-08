// =========================
// CONFIG
// =========================
const API_URL = (() => {
    const host = window.location.hostname;
    // [::1] must hit Flask locally — otherwise fetch goes to /.netlify/... and nothing updates.
    if (host === "127.0.0.1" || host === "localhost" || host === "::1") {
        return "http://127.0.0.1:5000";
    }
    // Netlify production/staging path
    return "/.netlify/functions/api";
})();
console.info("[MyVoiceGuard] API_URL =", API_URL);
const THRESHOLD_REAL = 90;   // >=90% REAL; API may send threshold_used (same value)
const THRESHOLD_FAKE = 90;

let selectedFile    = null;
let selectedURL     = "";
let selectedSource  = "";
let selectedMode    = "upload";   // "upload" | "link" | "live"
let mediaRecorder;
let audioChunks     = [];
let timerInterval;
let recognition;
let transcriptText  = "";    // speech-to-text result from live recording
let recordingSeconds = 0;
let analysisBusy = false;
const LAST_RESULT_KEY = "mvg_last_result_v1";
let lastResultMode = "upload"; // remembers source mode for Submit Another
const MVG_AGENT_LOG_ENABLED = (() => {
    try {
        const q = new URLSearchParams(window.location.search);
        if (q.get("mvgdebug") === "1") return true;
        return localStorage.getItem("mvgdebug") === "1";
    } catch (_) {
        return false;
    }
})();

// #region agent log
function agentLog(hypothesisId, location, message, data, runId = "pre-fix") {
    if (!MVG_AGENT_LOG_ENABLED) return;
    fetch("http://127.0.0.1:7761/ingest/9ba4045b-2532-49be-8464-d35216b0d93f", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "db268d" },
        body: JSON.stringify({
            sessionId: "db268d",
            hypothesisId,
            location,
            message,
            data,
            runId,
            timestamp: Date.now(),
        }),
    }).catch(() => {});
}
// #endregion

async function fetchWithApiFallback(path, options) {
    const candidates = [];
    const base = (API_URL || "").replace(/\/+$/, "");
    if (base) candidates.push(base);
    for (const alt of ["http://127.0.0.1:5000", "http://localhost:5000", "http://[::1]:5000"]) {
        if (!candidates.includes(alt)) candidates.push(alt);
    }

    let lastErr = null;
    for (const baseUrl of candidates) {
        try {
            const url = `${baseUrl}${path}`;
            agentLog("H_REQ", "fetchWithApiFallback:try", "api_try", { url, path });
            const res = await fetch(url, options);
            agentLog("H_REQ", "fetchWithApiFallback:ok", "api_ok", { url, status: res.status, ok: res.ok });
            return { res, usedBase: baseUrl };
        } catch (err) {
            lastErr = err;
            agentLog("H_REQ", "fetchWithApiFallback:err", "api_err", {
                baseUrl,
                path,
                message: err?.message || String(err),
                name: err?.name || null,
            });
        }
    }
    throw lastErr || new Error("All API endpoints failed");
}

// =========================
// BACKEND HEALTH CHECK
// =========================
async function checkBackend() {
    try {
        const res = await fetch(`${API_URL}/health`, {
            method: "GET",
            headers: { "Accept": "application/json" },
            cache: "no-store",
        });
        return res.ok;
    } catch (err) {
        console.warn("⚠️ checkBackend failed:", err?.message || err);
        return false;
    }
}

window.addEventListener("load", async () => {
    const ok = await checkBackend();
    console.log(ok ? "✅ Backend connected!" : "⚠️ Backend not running – run: python app.py");
    // Default behavior requested: refresh opens Home page.
    showPage("home");
});

// =========================
// PAGE NAVIGATION
// =========================
function showPage(id) {
    if (typeof window !== "undefined" && window.event && typeof window.event.preventDefault === "function") {
        window.event.preventDefault();
    }
    document.querySelectorAll(".page-section").forEach(p => p.classList.remove("active"));
    const page = document.getElementById(id);
    if (page) page.classList.add("active");
    if (id === "detection") setMode("upload");
}

// =========================
// MODE SWITCH
// =========================
function setMode(mode) {
    selectedMode = mode;
    ["upload", "link", "live", "ready", "analysis"].forEach(m => {
        const el = document.getElementById("ui-" + m);
        if (el) el.classList.add("hidden");
    });
    const result = document.getElementById("ui-result");
    if (result) result.classList.add("hidden");

    const active = document.getElementById("ui-" + mode);
    if (active) active.classList.remove("hidden");

    if (mode === "live") {
        document.getElementById("live-start-state").classList.remove("hidden");
        document.getElementById("live-active-state").classList.add("hidden");
        clearInterval(timerInterval);
        document.getElementById("timer").innerText = "00:00";
        transcriptText = "";
    }
}

function setAnalyzeBusyState(isBusy) {
    const btn = document.getElementById("analyzeNowBtn");
    if (!btn) return;
    btn.disabled = !!isBusy;
    btn.style.opacity = isBusy ? "0.6" : "1";
    btn.style.pointerEvents = isBusy ? "none" : "auto";
    btn.innerText = isBusy ? "Analyzing..." : "Analyze Now";
}

// =========================
// FILE UPLOAD
// =========================
function fileSelected(input) {
    const file = input.files[0];
    if (!file) return;

    if (file.size > 50 * 1024 * 1024) {
        alert("❌ File too large! Maximum size is 50MB.");
        input.value = "";
        return;
    }

    const ext = file.name.split(".").pop().toLowerCase();
    if (!["mp3", "wav", "ogg", "m4a"].includes(ext)) {
        alert("❌ Unsupported file type!\nAllowed: MP3, WAV, OGG, M4A");
        input.value = "";
        return;
    }

    selectedFile   = file;
    selectedURL    = "";
    selectedSource = `Upload: ${file.name}`;

    console.log("✅ File selected:", file.name);
    document.getElementById("ui-upload").classList.add("hidden");
    document.getElementById("ui-ready").classList.remove("hidden");
    document.getElementById("readyFileName").innerText = "FILE: " + file.name;
}

// Drag & Drop
document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("dropArea");
    if (!dropArea) return;
    dropArea.addEventListener("dragover",  e => { e.preventDefault(); dropArea.style.borderColor = "#ccff00"; });
    dropArea.addEventListener("dragleave", () => { dropArea.style.borderColor = ""; });
    dropArea.addEventListener("drop", e => {
        e.preventDefault();
        dropArea.style.borderColor = "";
        if (e.dataTransfer.files.length > 0) fileSelected({ files: e.dataTransfer.files, value: "" });
    });

    const urlInput = document.getElementById("urlInput");
    if (urlInput) urlInput.addEventListener("keypress", e => { if (e.key === "Enter") processLink(); });
});

// =========================
// URL INPUT
// =========================
function processLink() {
    const url = document.getElementById("urlInput").value.trim();
    if (!url) { alert("Please enter a valid audio URL."); return; }

    try { new URL(url); } catch (_) { alert("❌ Invalid URL format."); return; }

    selectedFile   = null;
    selectedURL    = url;
    selectedSource = `URL: ${url}`;

    console.log("✅ URL selected:", url);
    document.getElementById("ui-link").classList.add("hidden");
    document.getElementById("ui-ready").classList.remove("hidden");
    document.getElementById("readyFileName").innerText =
        "LINK: " + url.substring(0, 50) + (url.length > 50 ? "…" : "");
}

// =========================
// LIVE RECORDING
// =========================
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Pick best MIME type
        let mimeType = "audio/webm";
        for (const t of ["audio/ogg; codecs=opus", "audio/webm; codecs=opus", "audio/webm"]) {
            if (MediaRecorder.isTypeSupported(t)) { mimeType = t; break; }
        }
        console.log("🎙️ MIME type:", mimeType);

        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks   = [];
        transcriptText = "";

        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: mimeType });
            const ext  = mimeType.includes("ogg") ? "ogg" : "webm";
            selectedFile   = new File([blob], `recording.${ext}`, { type: mimeType });
            selectedURL    = "";
            selectedSource = "Live Recording";

            console.log("✅ Recording ready:", selectedFile.name,
                        `${(selectedFile.size / 1024).toFixed(1)}KB`);
            console.log("✅ Transcript:", transcriptText);

            document.getElementById("ui-live").classList.add("hidden");
            document.getElementById("ui-ready").classList.remove("hidden");
            document.getElementById("readyFileName").innerText =
                "LIVE RECORDING (" + formatTime(recordingSeconds) + ")";
        };

        mediaRecorder.start(1000);

        // Speech-to-text
        const SR = window.webkitSpeechRecognition || window.SpeechRecognition;
        if (SR) {
            recognition = new SR();
            recognition.continuous      = true;
            recognition.interimResults  = true;
            recognition.lang            = "ms-MY";
            recognition.onresult = event => {
                let interim = "";
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const t = event.results[i][0].transcript;
                    if (event.results[i].isFinal) transcriptText += t + " ";
                    else interim += t;
                }
                const liveText = document.getElementById("liveText");
                if (liveText) liveText.innerText = (transcriptText + interim).trim() || "Listening…";
            };
            recognition.onerror = e => console.warn("Speech recognition error:", e.error);
            recognition.start();
        }

        // Timer
        recordingSeconds = 0;
        timerInterval = setInterval(() => {
            recordingSeconds++;
            document.getElementById("timer").innerText = formatTime(recordingSeconds);
            if (recordingSeconds >= 60) stopRecording();
        }, 1000);

        document.getElementById("live-start-state").classList.add("hidden");
        document.getElementById("live-active-state").classList.remove("hidden");
        document.getElementById("liveText").innerText = "Listening…";

    } catch (err) {
        console.error("Microphone error:", err);
        alert(err.name === "NotAllowedError"
            ? "❌ Microphone access denied!\nAllow microphone access in browser settings."
            : "❌ Cannot access microphone: " + err.message);
    }
}

function formatTime(s) {
    return `${Math.floor(s/60).toString().padStart(2,"0")}:${(s%60).toString().padStart(2,"0")}`;
}

function stopRecording() {
    try {
        if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
        if (recognition) recognition.stop();
        clearInterval(timerInterval);
        console.log("✅ Recording stopped. Transcript:", transcriptText);
    } catch (err) {
        console.error("Stop error:", err);
    }
}

// =========================
// MAIN ANALYSIS
// =========================
async function runFinalAnalysis() {
    if (analysisBusy) {
        console.warn("⏳ Analysis already running; ignoring duplicate click.");
        return;
    }
    analysisBusy = true;
    setAnalyzeBusyState(true);
    console.log("🔍 Starting analysis | mode:", selectedMode);
    console.log("File:", selectedFile, "URL:", selectedURL);
    // #region agent log
    agentLog("H_REQ", "runFinalAnalysis:start", "analysis_start", {
        mode: selectedMode,
        apiUrl: API_URL,
        hasFile: selectedFile instanceof File,
        hasUrl: !!selectedURL,
    });
    // #endregion

    if (!(selectedFile instanceof File) && !selectedURL) {
        analysisBusy = false;
        setAnalyzeBusyState(false);
        alert("❌ No audio source!\nPlease upload a file, enter a URL, or record audio.");
        return;
    }
    try { sessionStorage.removeItem(LAST_RESULT_KEY); } catch (_) {}

    // Do not hard-block on preflight health checks.
    // We still attempt the real request; /health can fail transiently while API works.
    const backendOk = await checkBackend();
    if (!backendOk) {
        console.warn("⚠️ Health check failed, continuing with real request...");
        // #region agent log
        agentLog("H_REQ", "runFinalAnalysis:health", "health_failed_but_continue", {
            mode: selectedMode,
            apiUrl: API_URL,
        });
        // #endregion
    }

    // Show analysis UI and persist state so an unexpected reload restores this stage.
    setMode("analysis");

    const bar = document.getElementById("progress-bar");
    const txt = document.getElementById("progress-text");
    bar.style.width = "0%";
    txt.innerText   = "0%";

    let fakeProgress = 0;
    const fakeTimer = setInterval(() => {
        if (fakeProgress < 85) {
            fakeProgress = Math.min(fakeProgress + Math.random() * 3, 85);
            bar.style.width = fakeProgress + "%";
            txt.innerText   = Math.floor(fakeProgress) + "%";
        }
    }, 200);

    try {
        let response;
        const modeForRequest = selectedMode;

        // -------------------------------------------------------
        // LIVE RECORDING → /predict-live  (sends transcript too)
        // -------------------------------------------------------
        const fileNameLower = (selectedFile?.name || "").toLowerCase();
        const fileTypeLower = (selectedFile?.type || "").toLowerCase();
        const looksLikeLiveBlob =
            selectedFile instanceof File &&
            (selectedSource === "Live Recording" ||
             fileNameLower.endsWith(".webm") ||
             fileNameLower.endsWith(".ogg") ||
             fileTypeLower.includes("audio/webm") ||
             fileTypeLower.includes("opus"));

        if ((modeForRequest === "live" || looksLikeLiveBlob) && selectedFile instanceof File) {
            console.log("📤 Live mode → /predict-live | transcript:", transcriptText.substring(0, 80));
            agentLog("H_REQ", "runFinalAnalysis:live", "sending_predict_live", {
                apiUrl: API_URL,
                fileName: selectedFile?.name || null,
                fileSize: selectedFile?.size || 0,
            });

            const formData = new FormData();
            formData.append("file",       selectedFile, selectedFile.name);
            formData.append("transcript", transcriptText.trim());

            const tried = await fetchWithApiFallback("/predict-live", {
                method: "POST",
                body:   formData,
            });
            response = tried.res;
            lastResultMode = "live";

        // -------------------------------------------------------
        // FILE UPLOAD → /predict-file
        // -------------------------------------------------------
        } else if (selectedFile instanceof File) {
            console.log("📤 File mode → /predict-file |", selectedFile.name);
            agentLog("H_REQ", "runFinalAnalysis:file", "sending_predict_file", {
                apiUrl: API_URL,
                fileName: selectedFile?.name || null,
                fileSize: selectedFile?.size || 0,
                fileType: selectedFile?.type || null,
            });

            const formData = new FormData();
            formData.append("file", selectedFile, selectedFile.name);

            const tried = await fetchWithApiFallback("/predict-file", {
                method: "POST",
                body:   formData,
            });
            response = tried.res;
            lastResultMode = "upload";

        // -------------------------------------------------------
        // URL → /predict-url  (sends videoTitle hint if possible)
        // -------------------------------------------------------
        } else if (selectedURL) {
            console.log("🔗 URL mode → /predict-url |", selectedURL);
            agentLog("H_REQ", "runFinalAnalysis:url", "sending_predict_url", {
                apiUrl: API_URL,
                url: selectedURL.substring(0, 200),
            });

            // Try to extract video title from the page title if user has it open
            // (best-effort; works when user copies YouTube tab title into a hidden field)
            const videoTitle = document.getElementById("videoTitleHint")?.value || "";

            const tried = await fetchWithApiFallback("/predict-url", {
                method:  "POST",
                headers: { "Content-Type": "application/json", "Accept": "application/json" },
                body:    JSON.stringify({ url: selectedURL, videoTitle }),
            });
            response = tried.res;
            lastResultMode = "link";
        }

        clearInterval(fakeTimer);

        if (!response) throw new Error("No response from backend");
        agentLog("H_REQ", "runFinalAnalysis:response", "received_response", {
            status: response.status,
            ok: response.ok,
            mode: modeForRequest,
        });

        if (!response.ok) {
            const body = await response.text();
            throw new Error(`Backend ${response.status}: ${body}`);
        }

        const data = await response.json();
        console.log("✅ Backend result:", data);

        if (data.error) throw new Error("Backend error: " + data.error);

        bar.style.width = "100%";
        txt.innerText   = "100%";

        setTimeout(() => {
            try {
                showPage("result-page");
                document.getElementById("ui-analysis").classList.add("hidden");
                document.getElementById("ui-result").classList.remove("hidden");
                if (!data.source || data.source === "processed") data.source = selectedSource;
                try {
                    sessionStorage.setItem(
                        LAST_RESULT_KEY,
                        JSON.stringify({
                            ts: Date.now(),
                            source: selectedSource || "",
                            data,
                        })
                    );
                } catch (_) {}
                showResult(data);
            } catch (renderErr) {
                console.error("❌ Result render failed:", renderErr);
                alert("❌ Result render failed: " + (renderErr?.message || String(renderErr)));
            }
        }, 250);

    } catch (error) {
        clearInterval(fakeTimer);
        console.error("❌ Analysis error:", error);
        agentLog("H_REQ", "runFinalAnalysis:catch", "analysis_error", {
            mode: selectedMode,
            apiUrl: API_URL,
            name: error?.name || null,
            message: error?.message || String(error),
        });

        const analysisUi = document.getElementById("ui-analysis");
        if (analysisUi) analysisUi.classList.remove("hidden");

        let msg = "❌ Analysis failed!\n\n";
        if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
            msg += "Network error while calling API.\n";
            msg += `API URL: ${API_URL}\n`;
            msg += "1. Open terminal\n2. Run: python app.py\n3. Check: http://127.0.0.1:5000/health";
        } else {
            msg += error.message;
        }
        txt.innerText = "ERROR";
        bar.style.width = "100%";
        bar.style.background = "#ff4444";
        const p = document.getElementById("progress-text");
        if (p) p.innerText = "ERROR";
        alert(msg);
    } finally {
        analysisBusy = false;
        setAnalyzeBusyState(false);
    }
}

// =========================
// SHOW RESULT
// =========================
function showResult(data) {
    if (!data) { alert("❌ No data from backend."); return; }

    const confidence = parseFloat(data.confidence) || 0;
    const thr =
        typeof data.threshold_used === "number" ? data.threshold_used : THRESHOLD_REAL;
    const label =
        data.result === "REAL" || data.result === "FAKE"
            ? data.result
            : confidence >= thr
              ? "REAL"
              : "FAKE";
    let color;
    if (label === "REAL") color = "#00ff66";
    else color = "#ff4444";

    document.getElementById("accuracyText").innerHTML =
        `${confidence.toFixed(1)}%<br>Accuracy<br>Voice`;
    document.getElementById("timeText").innerHTML =
        `Time to detect:<br>${data.time || 0} Seconds`;
    document.getElementById("sourceText").innerText =
        data.source || selectedSource || "Unknown";

    const lbl = document.getElementById("deepfakeLabel");
    lbl.innerText    = label;
    lbl.style.color  = color;
    const headline = document.getElementById("resultHeadline");
    if (headline) {
        headline.innerText = label;
        headline.style.color = color;
    }

    loadPolitician(data.person || "unknown");
    if (data.insufficient_speech && data.speech_gate_message) {
        const info = document.getElementById("personInfo");
        if (info) {
            info.innerText = data.speech_gate_message;
        }
    }
    // #region agent log
    if (data.insufficient_speech) {
        fetch("http://127.0.0.1:7761/ingest/9ba4045b-2532-49be-8464-d35216b0d93f", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "db268d" },
            body: JSON.stringify({
                sessionId: "db268d",
                location: "script.js:showResult",
                message: "insufficient_speech_ui",
                data: { reason: data.speech_gate_reason || null },
                timestamp: Date.now(),
                hypothesisId: "H_UI",
                runId: "post-fix",
            }),
        }).catch(() => {});
    }
    // #endregion
    console.log("✅ Result:", { label, confidence, person: data.person });
}

// =========================
// POLITICIAN DATABASE
// =========================
function loadPolitician(name) {
    const db = {
        anwar: {
            img:  "Photo/Anwar.png",
            name: "Dato' Seri Anwar bin Ibrahim",
            info: `Perdana Menteri dan Menteri Kewangan Malaysia

• Tarikh Lahir: 10 Ogos 1947
• Tempat Lahir: Ceruk Tok Kun, Bukit Mertajam, Pulau Pinang
• Isteri: Dato' Seri Diraja Dr. Wan Azizah Wan Ismail
• Jawatan: Perdana Menteri & Menteri Kewangan
• Parti: PH  |  Parlimen: P063  |  Kawasan: Tambun
• No. Telefon: 03-88888000
• Email: anwaribrahim@pmo.gov.my`,
        },
        najib: {
            img:  "Photo/Najib.png",
            name: "Dato' Sri Najib Tun Razak",
            info: `Perdana Menteri Malaysia ke-6

• Tarikh Lahir: 23 Julai 1953
• Tempat Lahir: Kuala Lipis, Pahang
• Isteri: Datin Seri Rosmah Mansor
• Parti: UMNO / BN  |  Parlimen: P095  |  Kawasan: Pekan`,
        },
        mahathir: {
            img:  "Photo/Mahathir.png",
            name: "Tun Dr Mahathir bin Mohamad",
            info: `Perdana Menteri Malaysia ke-4 & ke-7

• Tarikh Lahir: 10 Julai 1925
• Tempat Lahir: Alor Setar, Kedah
• Isteri: Tun Dr Siti Hasmah Mohd Ali
• Parti: Pejuang  |  Parlimen: P202  |  Kawasan: Langkawi`,
        },
    };

    const p = db[name?.toLowerCase()];
    if (!p) {
        const detectedName = (name && name.toLowerCase() !== "unknown")
            ? name.replace(/\b\w/g, c => c.toUpperCase())
            : "Unknown Speaker";
        document.getElementById("personName").innerText  = detectedName;
        document.getElementById("personInfo").innerText  =
            "Speaker profile is not in the built-in card list.\nReference voice matching is still applied from /reference_voice/.";
        document.getElementById("personImage").src       = "";
        document.getElementById("personImage").style.display = "none";
        return;
    }

    document.getElementById("personImage").src          = p.img;
    document.getElementById("personImage").style.display = "block";
    document.getElementById("personName").innerText     = p.name;
    document.getElementById("personInfo").innerText     = p.info;
}

// =========================
// RESET
// =========================
function submitAnother() {
    selectedFile    = null;
    selectedURL     = "";
    selectedSource  = "";
    transcriptText  = "";
    try { sessionStorage.removeItem(LAST_RESULT_KEY); } catch (_) {}

    showPage("detection");
    document.getElementById("ui-result").classList.add("hidden");
    const fi = document.getElementById("audioFileInput"); if (fi) fi.value = "";
    const ui = document.getElementById("urlInput");        if (ui) ui.value = "";
    const targetMode = ["upload", "link", "live"].includes(lastResultMode) ? lastResultMode : "upload";
    setMode(targetMode);
    console.log("✅ Reset");
}

// =========================
// DOWNLOAD PDF
// =========================
async function downloadPDF() {
    const jsPDF = window.jspdf?.jsPDF;
    if (!jsPDF) {
        alert("❌ PDF library not loaded. Please refresh the page and try again.");
        return;
    }

    const getTxt = (id) => (document.getElementById(id)?.innerText || "").trim();
    const clean = (t) => String(t || "").replace(/\s+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    const confidence = clean(getTxt("accuracyText").replace(/Accuracy\s*Voice/i, "").trim());
    const detectTime = clean(getTxt("timeText").replace(/Time to detect:\s*/i, "").trim());
    const label = clean(getTxt("deepfakeLabel"));
    const person = clean(getTxt("personName"));
    const personInfo = clean(getTxt("personInfo"));
    const source = clean(getTxt("sourceText"));
    const verdict = (label || "").toUpperCase().includes("REAL") ? "REAL" : "FAKE";
    const logoSrc = document.querySelector("nav img")?.getAttribute("src") || "";
    const personImgSrc = document.getElementById("personImage")?.getAttribute("src") || "";

    const loadImageDataUrl = (src, opts = {}) => new Promise((resolve) => {
        if (!src) return resolve(null);
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            try {
                const c = document.createElement("canvas");
                c.width = img.naturalWidth || img.width;
                c.height = img.naturalHeight || img.height;
                const ctx = c.getContext("2d");
                if (!ctx) return resolve(null);
                ctx.drawImage(img, 0, 0);
                if (opts.removePurpleBg) {
                    const im = ctx.getImageData(0, 0, c.width, c.height);
                    const d = im.data;
                    for (let i = 0; i < d.length; i += 4) {
                        const r = d[i];
                        const g = d[i + 1];
                        const b = d[i + 2];
                        // Remove dark-purple backdrop behind logo.
                        const isDarkPurple =
                            r > 18 && r < 90 &&
                            g > 10 && g < 70 &&
                            b > 45 && b < 130 &&
                            b > r + 12 &&
                            b > g + 8;
                        if (isDarkPurple) {
                            d[i + 3] = 0;
                        }
                    }
                    ctx.putImageData(im, 0, 0);
                }
                resolve(c.toDataURL("image/png"));
            } catch (_) {
                resolve(null);
            }
        };
        img.onerror = () => resolve(null);
        img.src = src;
    });

    const [logoData, personData] = await Promise.all([
        loadImageDataUrl(logoSrc, { removePurpleBg: true }),
        loadImageDataUrl(personImgSrc),
    ]);

    const pdf = new jsPDF({ unit: "mm", format: "a4" });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 12;
    const contentW = pageW - margin * 2;
    let y = 14;

    // Header banner (logo style)
    pdf.setFillColor(58, 28, 113);
    pdf.roundedRect(margin, y, contentW, 24, 3, 3, "F");
    if (logoData) {
        // Draw logo directly (PNG alpha) so it blends with purple header.
        pdf.addImage(logoData, "PNG", margin + 4, y + 4, 58, 16);
    } else {
        pdf.setTextColor(255, 255, 255);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(16);
        pdf.text("MyVoiceGuard", margin + 6, y + 9);
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(10);
        pdf.text("AI Voice Verification Report", margin + 6, y + 16);
    }
    pdf.setFontSize(9);
    pdf.setTextColor(255, 255, 255);
    pdf.text(`Generated: ${new Date().toLocaleString()}`, pageW - margin - 6, y + 16, { align: "right" });
    y += 30;

    // Verdict badge
    const isReal = verdict === "REAL";
    const badgeColor = isReal ? [0, 180, 90] : [220, 60, 60];
    pdf.setFillColor(badgeColor[0], badgeColor[1], badgeColor[2]);
    pdf.roundedRect(margin, y, 40, 10, 2, 2, "F");
    pdf.setTextColor(255, 255, 255);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(11);
    pdf.text(`RESULT: ${verdict}`, margin + 20, y + 6.7, { align: "center" });
    y += 14;

    // Summary cards
    const cardW = (contentW - 8) / 2;
    const cardH = 18;
    const drawCard = (x, title, value) => {
        pdf.setDrawColor(210, 210, 210);
        pdf.setFillColor(247, 247, 252);
        pdf.roundedRect(x, y, cardW, cardH, 2, 2, "FD");
        pdf.setTextColor(70, 70, 70);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.text(title, x + 4, y + 6);
        pdf.setTextColor(20, 20, 20);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(12);
        pdf.text(value || "-", x + 4, y + 13);
    };
    drawCard(margin, "Confidence", confidence || "-");
    drawCard(margin + cardW + 8, "Time to detect", detectTime || "-");
    y += cardH + 8;

    const drawField = (title, value) => {
        pdf.setTextColor(40, 40, 40);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(10.5);
        pdf.text(title, margin, y);
        y += 5;
        pdf.setTextColor(30, 30, 30);
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(10);
        const lines = pdf.splitTextToSize(value || "-", contentW);
        pdf.text(lines, margin, y);
        y += (lines.length * 4.8) + 3;
    };

    drawField("Detected speaker", person || "Unknown Speaker");
    drawField("Deepfake detection", label || "-");
    drawField("Source audio", source || "-");

    // Speaker image section
    if (personData) {
        const imgW = 34;
        const imgH = 42;
        const imgX = margin;
        if (y + imgH + 8 < pageH - 22) {
            pdf.setTextColor(40, 40, 40);
            pdf.setFont("helvetica", "bold");
            pdf.setFontSize(10.5);
            pdf.text("Speaker image", margin, y);
            y += 4;
            pdf.setDrawColor(210, 210, 210);
            pdf.roundedRect(imgX, y, imgW, imgH, 2, 2, "D");
            pdf.addImage(personData, "PNG", imgX + 1, y + 1, imgW - 2, imgH - 2);
            y += imgH + 5;
        }
    }

    drawField("Speaker profile summary", personInfo || "-");

    // Footer
    pdf.setDrawColor(220, 220, 220);
    pdf.line(margin, pageH - 16, pageW - margin, pageH - 16);
    pdf.setTextColor(120, 120, 120);
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(8.5);
    pdf.text("MyVoiceGuard automated report", margin, pageH - 10);
    pdf.text("Confidential", pageW - margin, pageH - 10, { align: "right" });

    const safeName = (person || "unknown_speaker")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
    pdf.save(`myvoiceguard_report_${safeName}_${Date.now()}.pdf`);
}

// =========================
// SHARE
// =========================
function shareResult() {
    const confidence = document.getElementById("accuracyText")?.innerText?.split("\n")[0] || "";
    const label      = document.getElementById("deepfakeLabel")?.innerText || "";
    const person     = document.getElementById("personName")?.innerText    || "";
    const text       = `MyVoiceGuard:\n${person}\nAccuracy: ${confidence} | ${label}`;

    if (navigator.share) {
        navigator.share({ title: "MyVoiceGuard Result", text, url: location.href })
                 .catch(e => console.log("Share cancelled:", e));
    } else {
        navigator.clipboard?.writeText(text)
            .then(() => alert("✅ Copied to clipboard!"))
            .catch(() => alert("Share not supported.\n\n" + text));
    }
}