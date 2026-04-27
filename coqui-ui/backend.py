"""
Coqui TTS Local Backend Server
--------------------------------
Serves a REST API for the Coqui XTTS v2 model.
Run this first, then open index.html in your browser.

Usage:
    python backend.py
"""

import os
import uuid
import glob
import tempfile
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, expose_headers=["X-Output-Filename"])  # Allow requests from the HTML frontend

# Global TTS instance (loaded once on first use)
_tts = None

LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "it", "name": "Italian"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "pl", "name": "Polish"},
    {"code": "tr", "name": "Turkish"},
    {"code": "ru", "name": "Russian"},
    {"code": "nl", "name": "Dutch"},
    {"code": "cs", "name": "Czech"},
    {"code": "ar", "name": "Arabic"},
    {"code": "zh-cn", "name": "Chinese (Simplified)"},
    {"code": "hu", "name": "Hungarian"},
    {"code": "ko", "name": "Korean"},
    {"code": "ja", "name": "Japanese"},
    {"code": "hi", "name": "Hindi"},
]

TEMP_DIR = tempfile.gettempdir()
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")


def get_device():
    """Detect the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_tts():
    """Load the Coqui XTTS v2 model (lazy-loaded on first call)."""
    global _tts
    if _tts is None:
        import torch
        from TTS.api import TTS

        device = get_device()
        device_label = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple Silicon (MPS)", "cpu": "CPU"}.get(device, device)
        print(f"Loading Coqui XTTS v2 model on {device_label}...")
        print("(This may take a minute the first time — model downloads ~2 GB)")

        # Pre-accept the Coqui non-commercial license so it never prompts interactively
        os.environ["COQUI_TOS_AGREED"] = "1"

        use_gpu = (device == "cuda")
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

        # Attempt to move model to MPS for Apple Silicon acceleration
        if device == "mps":
            try:
                _tts.synthesizer.tts_model.to(device)
                print("✅ Apple Silicon MPS acceleration enabled — expect 3–5x speedup!")
            except Exception as e:
                print(f"⚠️  MPS acceleration unavailable for this model, falling back to CPU. ({e})")

        print("Model loaded successfully!")
    return _tts


@app.errorhandler(Exception)
def handle_exception(e):
    """Catch-all: always return JSON errors so the browser can display them."""
    print(f"Unhandled error: {e}")
    return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Coqui TTS backend is running"})


@app.route("/api/open-downloads", methods=["GET"])
def open_downloads():
    """Open the Downloads folder in Finder (macOS)."""
    try:
        subprocess.Popen(["open", DOWNLOADS_DIR])
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fetch-audio", methods=["POST"])
def fetch_audio():
    """
    Download a public audio/video URL and return a trimmed WAV clip.

    JSON body:
        url        (str)   - Public URL to audio or video
        start_time (float) - Start offset in seconds (default 0)
    """
    data = request.get_json() or {}
    url  = data.get("url", "").strip()
    start_time = float(data.get("start_time", 0))

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    clip_id     = uuid.uuid4().hex
    raw_tmpl    = os.path.join(TEMP_DIR, f"ytdl_{clip_id}.%(ext)s")
    clip_path   = os.path.join(TEMP_DIR, f"clip_{clip_id}.wav")

    try:
        # ── Step 1: download with yt-dlp ──────────────────────
        dl_cmd = [
            "yt-dlp",
            "--no-playlist",
            "-x",                          # extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", raw_tmpl,
            url,
        ]
        dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=120)
        if dl.returncode != 0:
            err = dl.stderr.strip().splitlines()[-1] if dl.stderr.strip() else "Download failed"
            return jsonify({"error": err}), 500

        # Find the downloaded file
        matches = glob.glob(os.path.join(TEMP_DIR, f"ytdl_{clip_id}.*"))
        if not matches:
            return jsonify({"error": "Downloaded file not found"}), 500
        raw_path = matches[0]

        # ── Step 2: trim with ffmpeg ──────────────────────────
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),        # seek to start time
            "-i", raw_path,
            "-t", "30",                    # extract up to 30 seconds
            "-ar", "22050",                # sample rate Coqui likes
            "-ac", "1",                    # mono
            clip_path,
        ]
        ff = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
        if ff.returncode != 0:
            return jsonify({"error": "ffmpeg trimming failed — is ffmpeg installed?"}), 500

        return send_file(clip_path, mimetype="audio/wav", as_attachment=False)

    except FileNotFoundError as e:
        tool = "yt-dlp" if "yt-dlp" in str(e) else "ffmpeg"
        return jsonify({"error": f"{tool} not found — run: brew install {tool}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Download timed out — try a shorter clip or different URL"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up raw download
        for f in glob.glob(os.path.join(TEMP_DIR, f"ytdl_{clip_id}.*")):
            try: os.unlink(f)
            except: pass


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Return list of supported languages."""
    return jsonify(LANGUAGES)


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    """
    Generate speech from text.

    Form fields:
        text        (str, required)  - The text to speak
        language    (str, optional)  - Language code (default: "en")
        speaker_wav (file, optional) - WAV audio file for voice cloning

    Returns:
        audio/wav file
    """
    text = request.form.get("text", "").strip()
    language = request.form.get("language", "en")
    speaker_file = request.files.get("speaker_wav")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 3000:
        return jsonify({"error": "Text too long. Please keep it under 3000 characters."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coqui_{timestamp}.wav"
    output_path = os.path.join(DOWNLOADS_DIR, filename)
    speaker_path = None

    try:
        tts = get_tts()  # inside try so errors return clean JSON
        if speaker_file and speaker_file.filename:
            # Voice cloning mode: use uploaded audio as speaker reference
            speaker_path = os.path.join(TEMP_DIR, f"speaker_{uuid.uuid4().hex}.wav")
            speaker_file.save(speaker_path)
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_path,
                language=language,
                file_path=output_path,
                gpt_cond_len=30,       # use up to 30s of reference audio for accent conditioning
                gpt_cond_chunk_len=4,  # finer chunks = better accent capture
            )
        else:
            # Standard mode: use default speaker
            tts.tts_to_file(
                text=text,
                language=language,
                file_path=output_path,
            )

        response = send_file(output_path, mimetype="audio/wav", as_attachment=False)
        response.headers["X-Output-Filename"] = filename
        return response

    except Exception as e:
        print(f"Error during synthesis: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary speaker file
        if speaker_path and os.path.exists(speaker_path):
            os.unlink(speaker_path)


if __name__ == "__main__":
    print("=" * 50)
    print("  Coqui TTS Backend Server")
    print("=" * 50)
    print("Starting on http://localhost:5000")
    print("Open index.html in your browser to use the UI.")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
