#!/usr/bin/env python3
"""
Squirrel Notes v1.0 🐿️
-------------------------------
AI meeting notes for people who were definitely paying attention.
Records Windows system audio, transcribes with
local OpenAI Whisper, and generates structured meeting summaries
using a local Ollama LLM.

Usage:
    1.  pip install -r requirements.txt
    2.  python app.py
    3.  Open http://localhost:5000
"""

import os
import sys
import json
import time
import threading
import traceback

import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    sd = sf = None
    HAS_AUDIO = False

try:
    import whisper as _whisper
    HAS_WHISPER = True
except ImportError:
    _whisper = None
    HAS_WHISPER = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    _requests = None
    HAS_REQUESTS = False

try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
    HAS_TK = True
except ImportError:
    _tk = _filedialog = None
    HAS_TK = False

SAMPLE_RATE = 16_000
CHANNELS    = 1

# ── Persistent config (saved to config.json next to app.py) ───────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

_DEV_MEETINGS_INSTRUCTIONS = """\
You are Chip, a highly disciplined senior software engineer squirrel with the attention span of a golden retriever at a tennis ball factory — except every single bit of it is laser-focused on this meeting. You have an almost pathological need to document everything. You have been sent to observe this meeting from a nearby oak tree with a tiny notepad. Your job is to collect every acorn of useful information and bury it in a structured, professional summary that the team can dig up later.

You document meetings covering Jira Issues, Stories and Epics, business requirements, requirement changes, brainstorming sessions, and technical challenges. These notes will be used to develop software solutions, so accuracy and technical detail are critical.

Squirrel Code of Conduct:
- ZERO hallucination. Squirrels of integrity do not bury fake nuts. Do not fabricate speaker names, dates, Jira numbers, table names, or tasks.
- Flag anything unclear with [unclear] or TBD. If you didn't hear it clearly from the tree, say so.
- Be concise but thorough. Every acorn matters.
- Make all section headers bold.

Output Format — follow this structure exactly:

**Topic(s):** [What was this meeting about?]
**Attendees:** [List all speakers/participants identified in the transcript]

---

**Key Discussion Points & Decisions**
1. [First main topic] — [Bullet points with details and decisions]
2. [Next topic] — [Continue as needed]

---

**Next Steps & Timeline**
[Bullet points: what happens next and by when, if mentioned]

---

**Jira Issues, Stories & Epics**
| Jira Number | Title / Subject | Additional Details |
|-------------|-----------------|-------------------|
| ...         | ...             | ...               |

---

**Action Items & Owners**
| Action Item | Owner | Due Date | Additional Details |
|-------------|-------|----------|--------------------|
| TODO: [item] | Christie | [date or TBD] | [notes] |
| ...          | ...   | ...      | ...                |
(Prefix any action item where Christie is the owner with "TODO:")

---

**Points of Contact**
| Name | Role / Department | Contact Context |
|------|-------------------|-----------------|
| ...  | ...               | ...             |

---

**Summary**
[One or two paragraphs wrapping up what was decided and what the overall outcome of the meeting was. Neutral, professional, human-sounding — Chip is thorough but he knows when to wrap it up.]

---

**Technical Summary**
[Summarize all technical items: server names, database names, stored procedures, filenames, function names, APIs, services, etc. Be thorough — this is the good stuff.]

**Tables & Columns**
| Table (camelCase) | Column (camelCase) | Details / FK Relationships |
|-------------------|--------------------|---------------------------|
| ...               | ...                | ...                       |

(All table and column names must be in camelCase — no underscores, no spaces. Note any foreign key relationships in the Details column.)

---

Instructions for Chip the Squirrel:
1. Read the full transcript carefully before writing anything. Every word is an acorn.
2. Identify all unique speakers. If names are missing, use roles (e.g., "Product Manager," "Developer 2").
3. Segment the transcript into discussion topics based on natural transitions and subject changes.
4. Highlight all decisions — look for phrases like "we agreed," "let's go with," "we'll move forward with," "we decided."
5. Extract all Jira numbers, titles, and any history or context discussed around them.
6. Convert every concrete task into an Action Item row. If Christie is the owner, prepend with TODO:.
7. Extract anyone mentioned as a point of contact, product owner, or subject matter expert.
8. In the Technical Summary, capture every technical detail mentioned: servers, databases, tables, columns, stored procedures, code files, APIs, services, and architectures.
9. All table/column names go in the Tables & Columns section in camelCase. Note FK relationships.
10. Close with a crisp, high-level Summary paragraph — the nut buried at the top of the cache.\
"""

_DEFAULT_CFG: dict = {
    "wmodel":                "base",
    "omodel":                "",
    "ollama_base":           "http://localhost:11434",
    "vault_path":            "",
    "audio_path":            "",
    "summary_path":          "",
    "raw_path":              "",
    "device_index":          None,
    "meeting_instructions":  _DEV_MEETINGS_INSTRUCTIONS,
}

def _load_cfg() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                return {**_DEFAULT_CFG, **json.load(fh)}
        except Exception:
            pass
    return dict(_DEFAULT_CFG)

def _save_cfg(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

def _ensure_dirs(cfg: dict):
    for key in ("audio_path", "summary_path", "raw_path"):
        try:
            os.makedirs(cfg[key], exist_ok=True)
        except Exception:
            pass

_cfg = _load_cfg()
_ensure_dirs(_cfg)

# Convenience accessors so the rest of the code reads cleanly
def AUDIO_PATH()   -> str: return _cfg["audio_path"]
def SUMMARY_PATH() -> str: return _cfg["summary_path"]
def RAW_PATH()     -> str: return _cfg["raw_path"]

# ── Application state ─────────────────────────────────────────────────────────
_lock         = threading.Lock()
_chunks: list = []
_stream       = None        # active sd.InputStream
_whisper_mdl  = None        # cached Whisper model

state: dict = {
    "status":          "idle",   # idle|recording|processing|transcribing|summarizing|done|error
    "message":         "Ready. Click \u25cf Start Recording to begin.",
    "audio_file":      None,
    "summary_file":    None,
    "transcript":      None,
    "summary":         None,
    "error":           None,
    "rec_start":       None,
    "duration":        0,
    "wmodel":          _cfg.get("wmodel", "base"),
    "omodel":          _cfg.get("omodel", "llama3:latest"),
    "raw_file":        None,     # path to the raw Whisper transcript .txt
    "meeting_context": None,     # populated from Outlook when user clicks Fetch
}

# ── Meeting-summary prompt ────────────────────────────────────────────────────
# ── LLM extraction prompts (simple key=value, one field at a time) ─────────────
#
# Strategy: Python owns the template 100%. The LLM is only asked short, specific
# questions that return a plain value or the word NONE. Python then assembles the
# final markdown, so the model cannot hallucinate structure or invent table rows.

EXTRACT_SYSTEM = (
    "You extract specific facts from a meeting transcript. "
    "Answer with only the requested information — no explanations, no examples. "
    "If the answer is not explicitly present in the transcript, respond with exactly: NONE"
)

# Each prompt is sent separately. {transcript} is replaced before sending.
EXTRACT_QUESTIONS = {
    "topics":      "What are the main topics discussed in this transcript? List them briefly, separated by semicolons. TRANSCRIPT:\n{transcript}",
    "speakers":    "What speaker names or labels are explicitly mentioned in this transcript? List them separated by semicolons. If no names appear, write NONE. TRANSCRIPT:\n{transcript}",
    "decisions":   "What decisions or agreements were explicitly stated (e.g. 'we agreed to...', 'let's go with...')? List each one separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "next_steps":  "What next steps or follow-up actions were explicitly mentioned? List each separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "jira":        "What Jira issue numbers, story names, or epic names were explicitly mentioned? Format each as 'number|title|details', separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "actions":     "What specific tasks were explicitly assigned to a named person? Format each as 'task|owner|due date or TBD', separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "contacts":    "What names were mentioned as a point of contact, product owner, or person to reach? Format each as 'name|role|details', separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "technical":   "What technical items were explicitly mentioned: server names, database names, stored procedures, filenames, function names? List each separated by semicolons. If none, write NONE. TRANSCRIPT:\n{transcript}",
    "tables":      "What database table or column names were explicitly mentioned? Format each as 'tableName|columnName|details' in camelCase. If none, write NONE. TRANSCRIPT:\n{transcript}",
}


def _parse_field(raw: str) -> list[str]:
    """Return a list of non-empty values, or empty list if the answer was NONE."""
    raw = raw.strip()
    if not raw or raw.upper() == "NONE" or raw.upper().startswith("NONE"):
        return []
    return [v.strip() for v in raw.split(";") if v.strip() and v.strip().upper() != "NONE"]


def _build_markdown(fields: dict, timestamp_str: str, meeting_context: dict = None) -> str:
    """
    Python builds the entire markdown document from extracted field values.
    The LLM never sees the template, so it cannot deviate from it.
    If meeting_context is provided (from Outlook), it overrides the LLM-extracted
    topic and attendees with accurate calendar data.
    """
    def none_row(cols=3):
        return "| None mentioned |" + " — |" * (cols - 1)

    def table_rows(items: list, cols: int = 3) -> list[str]:
        if not items:
            return [none_row(cols)]
        rows = []
        for item in items:
            parts = [p.strip() for p in item.split("|")]
            while len(parts) < cols:
                parts.append("—")
            rows.append("| " + " | ".join(parts[:cols]) + " |")
        return rows

    # Prefer Outlook calendar data over LLM extraction for topic and attendees
    if meeting_context:
        topics   = meeting_context.get("subject") or "Not determinable from transcript"
        speakers = ", ".join(meeting_context.get("attendees") or []) or "Not identifiable from transcript"
    else:
        topics   = "; ".join(fields["topics"])   or "Not determinable from transcript"
        speakers = "; ".join(fields["speakers"]) or "Not identifiable from transcript"
    decisions = fields["decisions"]
    steps     = fields["next_steps"]
    jira      = fields["jira"]
    actions   = fields["actions"]
    contacts  = fields["contacts"]
    technical = fields["technical"]
    tables    = fields["tables"]

    # Key discussion points: merge topics + decisions into numbered list
    discussion_items = list(fields["topics"])
    decision_notes   = list(decisions)

    md = []
    md.append(f"**Topic(s):** {topics}")
    md.append(f"**Attendees:** {speakers}")
    md.append("")

    md.append("## Key Discussion Points & Decisions")
    if not discussion_items:
        md.append("None mentioned.")
    else:
        for i, topic in enumerate(discussion_items, 1):
            md.append(f"{i}. {topic}")
            # Attach any decisions as sub-bullets under the first topic
            if i == 1 and decision_notes:
                for d in decision_notes:
                    md.append(f"   • Decision: {d}")
    md.append("")

    md.append("## Next Steps & Timeline")
    if not steps:
        md.append("• None mentioned.")
    else:
        for s in steps:
            md.append(f"• {s}")
    md.append("")

    md.append("## Jira Issues, Stories or Epics")
    md.append("| Jira Number | Jira Subject or Title | Additional Details |")
    md.append("|---|---|---|")
    md.extend(table_rows(jira, 3))
    md.append("")

    md.append("## Action Items & Owners")
    md.append("| Action Item | Owner | Additional Details |")
    md.append("|---|---|---|")
    if not actions:
        md.append(none_row(3))
    else:
        for item in actions:
            parts = [p.strip() for p in item.split("|")]
            while len(parts) < 3:
                parts.append("—")
            task, owner, detail = parts[0], parts[1], parts[2]
            todo = "**TODO** " if "christie" in owner.lower() else ""
            md.append(f"| {todo}{task} | {owner} | {detail} |")
    md.append("")

    md.append("## Points of Contact")
    md.append("| Point of Contact | Owner | Additional Details |")
    md.append("|---|---|---|")
    md.extend(table_rows(contacts, 3))
    md.append("")

    md.append("## Summary")
    md.append(f"• {topics}")
    if decisions:
        for d in decisions:
            md.append(f"• {d}")
    md.append("")

    md.append("## **Technical Summary:**")
    if not technical:
        md.append("• No technical details mentioned.")
    else:
        for t in technical:
            md.append(f"• {t}")
    md.append("")

    md.append("## Tables and Columns")
    md.append("| Table | Column | Details |")
    md.append("|---|---|---|")
    md.extend(table_rows(tables, 3))

    return "\n".join(md)

# ── Audio helpers ─────────────────────────────────────────────────────────────

def _audio_callback(indata, frames, time_info, status_flags):
    """Called by sounddevice for each audio block; appends to buffer."""
    _chunks.append(indata.copy())


def get_loopback_device():
    """
    Find a suitable Windows WASAPI loopback device.
    Returns (device_index_or_None, use_wasapi_flag).
    """
    if not HAS_AUDIO:
        return None, False

    # 1. Look for an already-labelled loopback input device (e.g. Stereo Mix)
    try:
        for i, dev in enumerate(sd.query_devices()):
            nm = dev.get("name", "").lower()
            if "loopback" in nm and dev.get("max_input_channels", 0) > 0:
                return i, False
    except Exception:
        pass

    # 2. Fall back: use the default output device + WasapiSettings(loopback=True)
    try:
        out_idx = sd.default.device[1]
        if isinstance(out_idx, (int, float)) and int(out_idx) != -1:
            return int(out_idx), True
    except Exception:
        pass

    return None, False


def _fix_device_name(name: str) -> str:
    """PortAudio truncates device names at 32 chars, often cutting off the closing ')'.
    If the name has an unmatched '(' we add '…)' to make it readable."""
    if name.count("(") > name.count(")"):
        name = name.rstrip() + "…)"
    return name


def list_audio_devices():
    """Return a list of available input devices."""
    if not HAS_AUDIO:
        return []
    result = []
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                result.append({"index": i, "name": _fix_device_name(dev["name"])})
    except Exception:
        pass
    return result


# ── Recording control ─────────────────────────────────────────────────────────

def start_capture(device_index=None):
    global _stream, _chunks
    if not HAS_AUDIO:
        raise RuntimeError(
            "sounddevice / soundfile are not installed. "
            "Run: pip install sounddevice soundfile"
        )

    _chunks = []

    # Resolve device: explicit arg > saved config > loopback auto-detect
    if device_index is None and _cfg.get("device_index") is not None:
        device_index = _cfg["device_index"]

    base_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_audio_callback,
    )

    if device_index is not None:
        # User chose a specific device — use it directly, no loopback magic needed
        base_kwargs["device"] = int(device_index)
        _stream = sd.InputStream(**base_kwargs)
        _stream.start()
        return

    # No device specified — try WASAPI loopback, fall back gracefully
    dev_idx, use_wasapi = get_loopback_device()
    if dev_idx is not None:
        base_kwargs["device"] = dev_idx

    if use_wasapi:
        try:
            loopback_kwargs = {**base_kwargs, "extra_settings": sd.WasapiSettings(loopback=True)}
            _stream = sd.InputStream(**loopback_kwargs)
            _stream.start()
            return
        except Exception:
            # WASAPI loopback not supported on this system — fall through
            pass

    # Final fallback: default input device
    _stream = sd.InputStream(**base_kwargs)
    _stream.start()


def stop_capture():
    global _stream
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None


def save_wav(timestamp_str: str) -> str:
    if not _chunks:
        raise RuntimeError("No audio data was captured.")
    audio = np.concatenate(_chunks, axis=0)
    path  = os.path.join(AUDIO_PATH(), f"{timestamp_str}.wav")
    sf.write(path, audio, SAMPLE_RATE)
    return path


# ── Transcription ─────────────────────────────────────────────────────────────

def _load_whisper(model_name: str):
    global _whisper_mdl
    if not HAS_WHISPER:
        raise RuntimeError(
            "openai-whisper is not installed. "
            "Run: pip install openai-whisper"
        )
    cached_name = getattr(_whisper_mdl, "_model_name", None)
    if _whisper_mdl is None or cached_name != model_name:
        _set_state(message=f"Loading Whisper '{model_name}' model — this may take a minute…")
        mdl = _whisper.load_model(model_name)
        mdl._model_name = model_name
        with _lock:
            _whisper_mdl = mdl


def transcribe_audio(audio_path: str, model_name: str) -> str:
    _load_whisper(model_name)
    _set_state(
        status="transcribing",
        message="Transcribing audio with Whisper — please wait…"
    )
    # Load audio as a numpy array via soundfile so ffmpeg is NOT required.
    # Whisper accepts a float32 mono array at 16 kHz directly.
    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:          # convert stereo → mono
        audio_np = audio_np.mean(axis=1)
    result = _whisper_mdl.transcribe(audio_np, fp16=False)
    return result["text"].strip()


# ── Summarisation ─────────────────────────────────────────────────────────────

def _ollama_chat(system: str, user: str, model: str, timeout: int = 120) -> str:
    """Single Ollama /api/chat call. Returns the assistant content string."""
    if not model or not model.strip():
        raise RuntimeError(
            "No Ollama model selected. Go to Settings → Ollama Model and choose a model, then save."
        )
    try:
        resp = _requests.post(
            f"{_cfg['ollama_base']}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
    except _requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running: open a terminal and run 'ollama serve'."
        )

    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{model}' not found. "
            f"Run 'ollama list' to see installed models, then update Settings."
        )

    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def generate_summary(transcript: str, ollama_model: str, timestamp_str: str = "", meeting_context: dict = None) -> str:
    """
    Hallucination-proof pipeline:
      - Ask the LLM one short, specific question per field (9 questions total).
      - Each answer is either a brief value or the word NONE.
      - Python assembles the final markdown — the LLM never sees or controls the template.

    This prevents hallucination because the model is doing simple text retrieval
    (find X in this passage), not open-ended generation into a blank template.
    """
    if not HAS_REQUESTS:
        raise RuntimeError("requests library is not installed.")

    total = len(EXTRACT_QUESTIONS)
    extracted = {}

    for i, (field, prompt_template) in enumerate(EXTRACT_QUESTIONS.items(), 1):
        _set_state(
            status="summarizing",
            message=f"Extracting field {i}/{total}: {field} ({ollama_model})…"
        )
        user_msg = prompt_template.replace("{transcript}", transcript)
        raw = _ollama_chat(_cfg["meeting_instructions"], user_msg, ollama_model)
        extracted[field] = _parse_field(raw)

    _set_state(status="summarizing", message="Building summary document…")
    return _build_markdown(extracted, timestamp_str, meeting_context)


def save_markdown(timestamp_str: str, summary: str) -> str:
    path = os.path.join(SUMMARY_PATH(), f"{timestamp_str}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    return path


def save_raw_transcript(timestamp_str: str, transcript: str, meeting_context: dict = None) -> str:
    """Save the raw Whisper transcript as a plain .txt file in the _raw folder."""
    path = os.path.join(RAW_PATH(), f"{timestamp_str}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        # Prepend meeting context header if available so the raw file is self-contained
        if meeting_context:
            fh.write(f"Meeting: {meeting_context.get('subject', '')}\n")
            fh.write(f"Date:    {meeting_context.get('start', '')}\n")
            attendees = ", ".join(meeting_context.get("attendees") or [])
            if attendees:
                fh.write(f"Attendees: {attendees}\n")
            fh.write("\n" + ("─" * 60) + "\n\n")
        fh.write(transcript)
    return path


# ── State helpers ─────────────────────────────────────────────────────────────

def _set_state(**kw):
    with _lock:
        state.update(kw)


# ── Background processing pipeline ───────────────────────────────────────────

def _pipeline(ts: str, wmodel: str, omodel: str, meeting_context: dict = None):
    """Runs in a daemon thread after recording stops."""
    try:
        # 1. Save WAV
        _set_state(status="processing", message="Saving audio file…")
        audio_path = save_wav(ts)
        _set_state(audio_file=audio_path)

        # 2. Transcribe
        transcript = transcribe_audio(audio_path, wmodel)

        # Save raw transcript immediately — available even if summarisation fails
        raw_path = save_raw_transcript(ts, transcript, meeting_context)
        _set_state(
            transcript=transcript,
            raw_file=raw_path,
            message="Transcription complete. Starting summary generation…"
        )

        # 3. Summarise (pass Outlook context so Python template uses it directly)
        summary = generate_summary(transcript, omodel, ts, meeting_context)

        # 4. Save markdown
        md_path = save_markdown(ts, summary)

        _set_state(
            status="done",
            summary=summary,
            summary_file=md_path,
            message="✅ Done! Summary saved to your Obsidian vault.",
        )

    except Exception as exc:
        _set_state(
            status="error",
            error=str(exc),
            message=f"⚠ Error: {exc}",
        )
        traceback.print_exc()


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/status")
def api_status():
    with _lock:
        s = dict(state)
    if s["status"] == "recording" and s["rec_start"]:
        s["duration"] = int(time.time() - s["rec_start"])
    return jsonify(s)


@app.route("/api/devices")
def api_devices():
    return jsonify(list_audio_devices())


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify({k: _cfg[k] for k in _DEFAULT_CFG})

    data = request.get_json(silent=True) or {}
    for key in _DEFAULT_CFG:
        if key in data:
            _cfg[key] = data[key]

    # Keep model state in sync
    _set_state(wmodel=_cfg["wmodel"], omodel=_cfg["omodel"])

    # Re-create any new directories
    _ensure_dirs(_cfg)
    _save_cfg(_cfg)
    return jsonify({"ok": True})


@app.route("/api/ollama-models")
def api_ollama_models():
    """Fetch the list of models installed in Ollama."""
    try:
        resp = _requests.get(f"{_cfg['ollama_base']}/api/tags", timeout=5)
        resp.raise_for_status()
        names = sorted(m["name"] for m in resp.json().get("models", []))
        return jsonify({"models": names})
    except Exception as exc:
        return jsonify({"models": [], "error": str(exc)})


@app.route("/api/readme")
def api_readme():
    """Serve the README.md as plain text for in-app rendering."""
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as fh:
            return jsonify({"content": fh.read()})
    except Exception as exc:
        return jsonify({"error": str(exc)})


@app.route("/api/browse-folder")
def api_browse_folder():
    """Open a native OS folder picker and return the chosen path."""
    if not HAS_TK:
        return jsonify({"error": "tkinter not available on this system"})
    try:
        root = _tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = _filedialog.askdirectory(title="Select base folder for Squirrel Notes")
        root.destroy()
        if folder:
            # Normalise to OS-native separators
            return jsonify({"path": os.path.normpath(folder)})
        return jsonify({"path": None})
    except Exception as exc:
        return jsonify({"error": str(exc)})


@app.route("/api/start", methods=["POST"])
def api_start():
    with _lock:
        if state["status"] == "recording":
            return jsonify({"error": "Already recording."}), 400

    data = request.get_json(silent=True) or {}
    dev  = data.get("device_index")  # None = auto-detect

    try:
        start_capture(dev)
        _set_state(
            status="recording",
            message="🔴 Recording… Click Stop when the meeting ends.",
            transcript=None,
            summary=None,
            error=None,
            rec_start=time.time(),
            duration=0,
            audio_file=None,
            summary_file=None,
        )
        return jsonify({"ok": True})
    except Exception as exc:
        _set_state(status="error", error=str(exc), message=f"⚠ {exc}")
        return jsonify({"error": str(exc)}), 500




@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        if state["status"] != "recording":
            return jsonify({"error": "Not currently recording."}), 400
        wmodel          = state["wmodel"]
        omodel          = state["omodel"]
        meeting_context = state.get("meeting_context")

    stop_capture()
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    _set_state(status="processing", message="Stopped. Saving and processing…", duration=0)

    t = threading.Thread(target=_pipeline, args=(ts, wmodel, omodel, meeting_context), daemon=True)
    t.start()
    return jsonify({"ok": True, "timestamp": ts})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _set_state(
        status="idle",
        message="Ready. Click \u25cf Start Recording to begin.",
        transcript=None,
        summary=None,
        error=None,
        rec_start=None,
        duration=0,
        audio_file=None,
        summary_file=None,
        raw_file=None,
        meeting_context=None,
    )
    return jsonify({"ok": True})


@app.route("/api/recent")
def api_recent():
    """List the 10 most recent recordings, summaries, and raw transcripts."""
    def _ls(path, ext):
        try:
            files = [f for f in os.listdir(path) if f.endswith(ext)]
            files.sort(reverse=True)
            return [{"name": f, "path": os.path.join(path, f)} for f in files[:10]]
        except Exception:
            return []

    return jsonify({
        "audio":      _ls(AUDIO_PATH(),   ".wav"),
        "summaries":  _ls(SUMMARY_PATH(), ".md"),
        "raw":        _ls(RAW_PATH(),     ".txt"),
    })


@app.route("/api/file-content")
def api_file_content():
    """Return the text content of a .md or .txt file."""
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found."}), 404
    if not path.lower().endswith((".md", ".txt")):
        return jsonify({"error": "Only .md and .txt files may be previewed."}), 403
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return jsonify({"content": fh.read()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/open-file")
def api_open_file():
    """Open a file in the OS default application (Windows: os.startfile)."""
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "File not found."}), 404
    try:
        os.startfile(path)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Inline HTML frontend ──────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Squirrel Notes 🐿️</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh}

    /* ── Design system ── */
    @keyframes gradientShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
    .tech-gradient{
      background:linear-gradient(135deg,#a855f7,#06b6d4,#a855f7);
      background-size:200% 200%;
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
      animation:gradientShift 5s ease infinite
    }
    .tech-gradient-static{
      background:linear-gradient(135deg,#a855f7,#06b6d4);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
    }
    .purple{color:#a855f7}.blue{color:#06b6d4}.muted{color:#6b7280}

    /* ── Navbar ── */
    .navbar{
      position:sticky;top:0;z-index:50;
      background:rgba(13,17,23,.82);
      backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
      border-bottom:1px solid rgba(255,255,255,.06);
      padding:.75rem 1.5rem;display:flex;align-items:center;gap:.75rem;
      transition:background .25s,border-color .25s,box-shadow .25s
    }
    .navbar.scrolled{
      background:rgba(13,17,23,.97);
      border-bottom-color:rgba(124,58,237,.22);
      box-shadow:0 1px 24px rgba(0,0,0,.45)
    }
    .navbar h1{font-size:1.1rem;font-weight:700;letter-spacing:-.01em}

    /* ── Badge ── */
    .badge{
      font-size:.68rem;font-weight:500;
      background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(8,145,178,.15));
      color:#a78bfa;border:1px solid rgba(124,58,237,.28);
      padding:2px 10px;border-radius:9999px;white-space:nowrap
    }

    /* ── Buttons ── */
    .btn{
      display:inline-flex;align-items:center;gap:.45rem;
      padding:.5rem 1.2rem;border-radius:9999px;
      font-size:.85rem;font-weight:600;cursor:pointer;
      transition:all .2s;border:none;white-space:nowrap;line-height:1;
      text-decoration:none;font-family:inherit
    }
    .btn-primary{
      background:linear-gradient(135deg,#7c3aed,#0891b2);
      color:#fff;box-shadow:0 0 0 0 rgba(124,58,237,0)
    }
    .btn-primary:hover{
      transform:translateY(-1px);
      box-shadow:0 4px 20px rgba(124,58,237,.4);filter:brightness(1.1)
    }
    .btn-primary:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none;filter:none}
    .btn-outline{
      background:transparent;border:1px solid #30363d;color:#8b949e
    }
    .btn-outline:hover{border-color:#a855f7;color:#a855f7;background:rgba(168,85,247,.06)}
    .btn-ghost{
      background:transparent;border:1px solid transparent;color:#6b7280
    }
    .btn-ghost:hover{color:#c9d1d9;background:rgba(255,255,255,.05)}
    .btn-danger{background:#b62324;color:#fff}
    .btn-danger:hover{background:#da3633;transform:translateY(-1px)}

    /* ── Nav pill (tabs) ── */
    .nav-pill{
      display:flex;align-items:center;
      background:rgba(22,27,34,.8);border:1px solid #21262d;
      border-radius:9999px;padding:.2rem;gap:.1rem;flex-wrap:wrap
    }
    .nav-item{
      padding:.32rem .85rem;border-radius:9999px;
      font-size:.8rem;font-weight:500;color:#6b7280;
      cursor:pointer;transition:all .15s;user-select:none;border:1px solid transparent
    }
    .nav-item:hover{color:#c9d1d9}
    .nav-item.active{
      background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(8,145,178,.2));
      border-color:rgba(124,58,237,.3);color:#c4b5fd
    }

    /* ── Layout ── */
    .main{max-width:1000px;margin:0 auto;padding:1.5rem 1rem}

    /* ── Outlook meeting card ── */
    .meeting-card{
      background:#161b22;border:1px solid #21262d;border-radius:10px;
      padding:1rem 1.25rem;margin-bottom:1rem;display:flex;
      align-items:flex-start;gap:1rem
    }
    .meeting-card.loaded{border-color:#1f6feb}
    .meeting-info{flex:1;min-width:0}
    .meeting-subject{font-size:.95rem;font-weight:600;color:#f0f6fc;
                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .meeting-meta{font-size:.78rem;color:#8b949e;margin-top:.25rem}
    .btn-outlook{
      display:inline-flex;align-items:center;gap:.4rem;padding:.45rem 1rem;
      border-radius:6px;border:1px solid #30363d;background:transparent;
      color:#8b949e;font-size:.8rem;cursor:pointer;transition:all .15s;white-space:nowrap
    }
    .btn-outlook:hover{background:#1f6feb22;border-color:#1f6feb;color:#58a6ff}
    .btn-outlook.loaded{border-color:#1f6feb;color:#58a6ff}
    .btn-clear-meeting{
      font-size:.7rem;color:#484f58;background:none;border:none;
      cursor:pointer;margin-left:.5rem;text-decoration:underline
    }
    .btn-clear-meeting:hover{color:#8b949e}

    /* ── Recording card ── */
    .rec-card{
      background:#161b22;border:1px solid #21262d;border-radius:14px;
      padding:2rem;text-align:center;margin-bottom:1.5rem;
      transition:border-color .2s,box-shadow .2s
    }
    .rec-card:hover{
      border-color:rgba(124,58,237,.22);
      box-shadow:0 0 0 1px rgba(124,58,237,.06),0 4px 24px rgba(0,0,0,.3)
    }
    .timer{font-size:3rem;font-weight:700;font-variant-numeric:tabular-nums;
           letter-spacing:2px;color:#f0f6fc;margin-bottom:.5rem}
    .timer.live{
      background:linear-gradient(135deg,#f85149,#ff9580);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
    }

    .status-row{display:flex;align-items:center;justify-content:center;
                gap:.5rem;margin-bottom:1.5rem;font-size:.875rem;color:#8b949e}
    .dot{width:8px;height:8px;border-radius:50%;background:#3fb950;flex-shrink:0}
    .dot.live{background:#f85149;animation:blink 1s infinite}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}

    .btn-rec{
      display:inline-flex;align-items:center;gap:.6rem;
      padding:.7rem 2rem;border-radius:9999px;border:none;
      font-size:1rem;font-weight:700;cursor:pointer;
      background:linear-gradient(135deg,#7c3aed,#0891b2);
      color:#fff;transition:all .2s;
      box-shadow:0 0 0 0 rgba(124,58,237,0)
    }
    .btn-rec:hover{
      transform:translateY(-1px);
      box-shadow:0 4px 20px rgba(124,58,237,.4);filter:brightness(1.1)
    }
    .btn-rec.live{
      background:linear-gradient(135deg,#b62324,#da3633)
    }
    .btn-rec.live:hover{
      box-shadow:0 4px 20px rgba(218,54,51,.4)
    }
    .btn-rec:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none;filter:none}

    .btn-secondary{
      margin-left:.75rem;padding:.7rem 1.25rem;border-radius:9999px;
      border:1px solid #30363d;background:transparent;color:#6b7280;
      font-size:.875rem;cursor:pointer;font-family:inherit;
      transition:all .15s
    }
    .btn-secondary:hover{border-color:#a855f7;color:#a855f7;background:rgba(168,85,247,.06)}

    .progress-msg{margin-top:1rem;font-size:.8rem;color:#6b7280;min-height:1.25em}

    /* ── Error banner ── */
    .err-banner{
      background:#3d0f0f;border:1px solid #6e1313;border-radius:8px;
      padding:.875rem 1rem;color:#ff7b72;font-size:.85rem;margin-bottom:1rem;display:none
    }

    /* ── Tabs (now nav-pill — keep .tabs wrapper for spacing) ── */
    .tabs{margin-bottom:1.25rem;display:flex}

    .panel{display:none}
    .panel.active{display:block}

    /* ── Content cards ── */
    .card{
      background:#161b22;border:1px solid #21262d;border-radius:10px;
      padding:1.25rem;transition:border-color .2s,box-shadow .2s
    }
    .card:hover{
      border-color:rgba(124,58,237,.22);
      box-shadow:0 0 0 1px rgba(124,58,237,.06),0 4px 24px rgba(0,0,0,.28)
    }
    .placeholder{color:#484f58;text-align:center;padding:3rem 1rem;font-size:.85rem}

    pre.transcript{
      white-space:pre-wrap;word-break:break-word;font-size:.82rem;
      line-height:1.7;color:#c9d1d9;font-family:inherit
    }

    /* ── Markdown output ── */
    .md h1,.md h2,.md h3{color:#f0f6fc;margin:1.25rem 0 .5rem}
    .md h1{font-size:1.15rem}.md h2{font-size:1rem}.md h3{font-size:.9rem}
    .md p{font-size:.85rem;line-height:1.7;color:#c9d1d9;margin:.4rem 0}
    .md ul,.md ol{padding-left:1.5rem;margin:.4rem 0}
    .md li{font-size:.85rem;line-height:1.6;color:#c9d1d9;margin:.15rem 0}
    .md strong{color:#f0f6fc}
    .md table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.8rem}
    .md th{background:#0d1117;padding:.5rem .75rem;text-align:left;
            color:#8b949e;border:1px solid #30363d}
    .md td{padding:.45rem .75rem;border:1px solid #21262d;
           color:#c9d1d9;vertical-align:top}
    .md a{color:#58a6ff;text-decoration:none}
    .md a:hover{text-decoration:underline}
    .md code{background:#0d1117;padding:1px 5px;border-radius:3px;font-size:.8rem}
    .md pre{background:#0d1117;padding:.75rem;border-radius:6px;
            overflow-x:auto;margin:.5rem 0}

    /* ── File link ── */
    .file-link{
      display:inline-flex;align-items:center;gap:.4rem;
      font-size:.75rem;color:#58a6ff;margin-top:.875rem;
      background:#0d1117;padding:.3rem .75rem;border-radius:5px;
      border:1px solid #21262d
    }

    /* ── Settings ── */
    .settings-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    .field{display:flex;flex-direction:column;gap:.35rem}
    .field label{font-size:.78rem;color:#8b949e;font-weight:500;text-transform:uppercase;letter-spacing:.04em}
    .field select,.field input{
      background:#0d1117;border:1px solid #30363d;border-radius:6px;
      color:#e2e8f0;padding:.45rem .75rem;font-size:.875rem;width:100%
    }
    .field select:focus,.field input:focus{outline:none;border-color:#58a6ff}
    .btn-save{
      margin-top:1rem;padding:.5rem 1.25rem;
      background:linear-gradient(135deg,#7c3aed,#0891b2);
      color:#fff;border:none;border-radius:9999px;
      cursor:pointer;font-size:.875rem;font-weight:600;
      font-family:inherit;transition:all .2s
    }
    .btn-save:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(124,58,237,.35);filter:brightness(1.1)}
    .help-txt{margin-top:.875rem;font-size:.78rem;color:#484f58;line-height:1.5}

    /* ── Recent files table ── */
    .recent-table{width:100%;border-collapse:collapse}
    .recent-table thead tr{border-bottom:2px solid #21262d}
    .recent-table th{
      font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;
      padding:.5rem .6rem;text-align:center;font-weight:500
    }
    .recent-table th:first-child{text-align:left}
    .recent-table tbody tr{border-bottom:1px solid #1a1f27;transition:background .1s}
    .recent-table tbody tr:hover{background:#161b22}
    .recent-name{
      font-size:.82rem;color:#c9d1d9;padding:.45rem .6rem;
      max-width:340px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis
    }
    .icon-btn{
      background:none;border:none;cursor:pointer;font-size:1.05rem;
      padding:.2rem .35rem;border-radius:5px;color:#484f58;
      transition:color .15s,background .15s;text-decoration:none;
      display:inline-flex;align-items:center;justify-content:center;line-height:1
    }
    .icon-btn:hover{color:#c9d1d9;background:#21262d}
    .icon-btn.ob-btn svg{transition:fill .15s}
    .icon-btn.ob-btn:hover svg{fill:#a78bfa}
    .icon-btn.disabled{opacity:.2;cursor:default;pointer-events:none}

    /* ── Inline file preview panel ── */
    .preview-panel{
      background:#0d1117;border:1px solid #1f6feb;border-radius:8px;
      padding:1.25rem;margin-top:1rem;display:none
    }
    .preview-panel.open{display:block}
    .preview-header{display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:.75rem}
    .preview-title{font-size:.8rem;color:#58a6ff;font-weight:600}
    .btn-close-preview{background:none;border:none;color:#484f58;font-size:1rem;
                       cursor:pointer;padding:0 .25rem}
    .btn-close-preview:hover{color:#8b949e}
  </style>
</head>
<body>
<header class="navbar" id="navbar">
  <span style="font-size:1.3rem;line-height:1">🐿️</span>
  <h1 class="tech-gradient">Squirrel Notes</h1>
  <span class="badge">Whisper &bull; Ollama &bull; Obsidian</span>
</header>

<div class="main">
  <div id="errBanner" class="err-banner"></div>

  <!-- ── Recording card ── -->
  <div class="rec-card">
    <div id="timer" class="timer">00:00:00</div>
    <div class="status-row">
      <div id="dot" class="dot"></div>
      <span id="statusTxt">Idle</span>
    </div>
    <div>
      <button id="btnRec" class="btn-rec" onclick="toggleRec()">
        <svg id="btnIcon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="12" cy="12" r="8"/>
        </svg>
        <span id="btnLabel">Start Recording</span>
      </button>
      <button class="btn-secondary" onclick="resetAll()">&#10227; Reset</button>
    </div>
    <div id="progressMsg" class="progress-msg">Ready. Click Start Recording to begin.</div>
  </div>

  <!-- ── Tabs ── -->
  <div class="tabs">
    <div class="nav-pill">
      <div class="nav-item active" onclick="showTab('transcript',this)">Transcript</div>
      <div class="nav-item"        onclick="showTab('summary',this)">Summary</div>
      <div class="nav-item"        onclick="showTab('settings',this)">Settings</div>
      <div class="nav-item"        onclick="showTab('recent',this)">Recent Files</div>
      <div class="nav-item"        onclick="showTab('readme',this)">About</div>
    </div>
  </div>

  <!-- ── Transcript panel ── -->
  <div id="panel-transcript" class="panel active">
    <div class="card">
      <div id="transcriptContent" class="placeholder">
        Transcript will appear here after processing.
      </div>
    </div>
  </div>

  <!-- ── Summary panel ── -->
  <div id="panel-summary" class="panel">
    <div class="card">
      <div id="summaryContent" class="md placeholder">
        Structured summary will appear here after processing.
      </div>
      <div id="summaryFileLink"></div>
    </div>
  </div>

  <!-- ── Settings panel ── -->
  <div id="panel-settings" class="panel">
    <div class="card">

      <p style="font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.75rem">AI Models</p>
      <div class="settings-grid">
        <div class="field">
          <label>Whisper Model</label>
          <select id="wmodel">
            <option value="tiny">tiny — fastest, lower accuracy</option>
            <option value="base">base — recommended balance</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
            <option value="large">large — slowest, highest accuracy</option>
          </select>
        </div>
        <div class="field">
          <label>Ollama Model
            <span id="ollamaLoadingBadge" style="color:#484f58;font-size:.7rem;font-weight:400;margin-left:.4rem">loading…</span>
          </label>
          <div style="display:flex;gap:.5rem;align-items:center">
            <select id="omodel" style="flex:1">
              <option value="">-- select a model --</option>
            </select>
            <button onclick="loadOllamaModels(document.getElementById('omodel').value)"
                    title="Refresh model list from Ollama"
                    style="padding:.3rem .75rem;font-size:.75rem;background:transparent;border:1px solid #30363d;color:#8b949e;border-radius:9999px;cursor:pointer;white-space:nowrap;font-family:inherit;transition:all .15s"
                    onmouseover="this.style.borderColor='#a855f7';this.style.color='#a855f7'"
                    onmouseout="this.style.borderColor='#30363d';this.style.color='#8b949e'">↻ Refresh</button>
          </div>
        </div>
      </div>

      <div class="settings-grid" style="grid-template-columns:1fr;margin-top:.75rem">
        <div class="field">
          <label>Ollama Base URL</label>
          <input id="ollama_base" type="text" placeholder="http://localhost:11434">
          <span style="font-size:.72rem;color:#484f58;margin-top:.3rem;display:block">
            Default is <code>http://localhost:11434</code>. Change only if Ollama runs on a different host or port.
          </span>
        </div>
      </div>

      <p style="font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin:.875rem 0 .75rem">Audio Capture</p>
      <div class="settings-grid">
        <div class="field" style="grid-column:span 2">
          <label>Input Device</label>
          <select id="devSel">
            <option value="">Default Input Device (auto)</option>
          </select>
          <span style="font-size:.72rem;color:#484f58;margin-top:.3rem;display:block">
            Select the microphone or webcam you want to record from. Settings are saved across sessions.
          </span>
        </div>
      </div>

      <p style="font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin:.875rem 0 .75rem">Folder Paths</p>
      <div class="settings-grid" style="grid-template-columns:1fr">
        <div class="field">
          <label>Obsidian Vault</label>
          <div style="display:flex;gap:.5rem;align-items:center">
            <input id="vault_path" type="text" placeholder="Click Browse to select your Obsidian vault…" style="flex:1" oninput="autoFillSubPaths()">
            <button onclick="browseFolder()"
                    style="padding:.35rem .9rem;font-size:.8rem;background:linear-gradient(135deg,#7c3aed,#0891b2);border:none;color:#fff;border-radius:9999px;cursor:pointer;white-space:nowrap;font-weight:600;font-family:inherit;transition:all .2s"
                    onmouseover="this.style.filter='brightness(1.1)';this.style.transform='translateY(-1px)'"
                    onmouseout="this.style.filter='';this.style.transform=''">
              📁 Browse
            </button>
          </div>
          <span style="font-size:.72rem;color:#484f58;margin-top:.3rem;display:block">
            A <code>Squirrel Notes</code> folder will be created inside your vault with audio, summaries, and raw sub-folders.
          </span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.75rem">
          <div class="field">
            <label style="font-size:.72rem">Audio sub-folder</label>
            <input id="audio_path" type="text" placeholder="audio" style="font-size:.8rem">
          </div>
          <div class="field">
            <label style="font-size:.72rem">Summaries sub-folder</label>
            <input id="summary_path" type="text" placeholder="summaries" style="font-size:.8rem">
          </div>
          <div class="field">
            <label style="font-size:.72rem">Raw transcripts sub-folder</label>
            <input id="raw_path" type="text" placeholder="raw" style="font-size:.8rem">
          </div>
        </div>
      </div>

      <p style="font-size:.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin:.875rem 0 .75rem">Meeting Instructions</p>
      <div class="settings-grid" style="grid-template-columns:1fr">
        <div class="field">
          <label>Summary Prompt / Instructions
            <span style="font-size:.7rem;color:#484f58;font-weight:400;margin-left:.4rem">— tells the AI what kind of meeting this is</span>
          </label>
          <textarea id="meeting_instructions" rows="7"
            style="width:100%;background:#010409;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;padding:.5rem .6rem;font-size:.8rem;font-family:inherit;resize:vertical;line-height:1.5"
            placeholder="Describe the meeting type and what to focus on…"></textarea>
          <div style="display:flex;align-items:center;gap:.75rem;margin-top:.35rem">
            <span style="font-size:.72rem;color:#484f58">
              ⚠ Saving overwrites this permanently — copy it somewhere first if you want to keep it.
            </span>
            <button onclick="resetInstructions()" style="margin-left:auto;font-size:.7rem;padding:.2rem .65rem;background:transparent;border:1px solid #30363d;color:#8b949e;border-radius:9999px;cursor:pointer;font-family:inherit;transition:all .15s" title="Reset to the Chip the Squirrel default"
                    onmouseover="this.style.borderColor='#a855f7';this.style.color='#a855f7'"
                    onmouseout="this.style.borderColor='#30363d';this.style.color='#8b949e'">↺ Reset to Default</button>
          </div>
        </div>
      </div>

      <button class="btn-save" onclick="saveSettings()" style="margin-top:1rem">Save Settings</button>
      <span id="saveConfirm" style="margin-left:.75rem;font-size:.8rem;color:#3fb950;display:none">✓ Saved!</span>

      <div class="help-txt" style="margin-top:1rem">
        <strong>Whisper models</strong> download automatically on first use (~150 MB for base).<br>
        <strong>Ollama</strong> must be running — open a terminal and run <code>ollama serve</code>.<br>
        <strong>Folders</strong> are created automatically inside your chosen base folder.
      </div>
    </div>
  </div>

  <!-- ── Recent files panel ── -->
  <div id="panel-recent" class="panel">
    <div class="card">
      <div id="recentContent" class="placeholder">Loading…</div>
      <!-- inline preview panel, shown when a file is clicked -->
      <div id="previewPanel" class="preview-panel">
        <div class="preview-header">
          <span id="previewTitle" class="preview-title"></span>
          <button class="btn-close-preview" onclick="closePreview()">&#10005;</button>
        </div>
        <div id="previewBody" class="md"></div>
      </div>
    </div>
  </div>

  <!-- ── About / README panel ── -->
  <div id="panel-readme" class="panel">
    <div class="card">
      <div id="readmeContent" class="md placeholder">Loading…</div>
    </div>
  </div>
</div>

<script>
  // ── State ──
  let isRecording    = false;
  let lastStatus     = 'idle';

  // ── Helpers ──
  function fmt(secs) {
    const h = String(Math.floor(secs / 3600)).padStart(2,'0');
    const m = String(Math.floor((secs % 3600) / 60)).padStart(2,'0');
    const s = String(secs % 60).padStart(2,'0');
    return h + ':' + m + ':' + s;
  }

  // ── Outlook meeting fetch ──
  function showErr(msg) {
    const b = document.getElementById('errBanner');
    b.textContent = '\u26A0 ' + msg;
    b.style.display = 'block';
  }
  function clearErr() {
    document.getElementById('errBanner').style.display = 'none';
  }

  function showTab(name, el) {
    document.querySelectorAll('.nav-item').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('panel-' + name).classList.add('active');
    if (name === 'recent') loadRecent();
    if (name === 'readme') loadReadme();
  }

  function loadReadme() {
    const el = document.getElementById('readmeContent');
    if (el.dataset.loaded) return; // only fetch once
    fetch('/api/readme').then(r => r.json()).then(d => {
      if (d.error) { el.textContent = 'Could not load README: ' + d.error; return; }
      el.className = 'md';
      el.innerHTML = marked.parse(d.content);
      el.dataset.loaded = '1';
    });
  }

  // ── API calls ──
  function toggleRec() {
    isRecording ? stopRec() : startRec();
  }

  function startRec() {
    clearErr();
    const devVal = document.getElementById('devSel').value;
    const body   = devVal !== '' ? { device_index: parseInt(devVal) } : {};
    fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(r => r.json()).then(d => {
      if (d.error) showErr(d.error);
    }).catch(e => showErr('Connection error: ' + e));
  }

  function stopRec() {
    fetch('/api/stop', { method: 'POST' })
      .then(r => r.json())
      .then(d => { if (d.error) showErr(d.error); })
      .catch(e => showErr('Connection error: ' + e));
  }

  function resetAll() {
    fetch('/api/reset', { method: 'POST' }).then(() => {
      document.getElementById('transcriptContent').className = 'placeholder';
      document.getElementById('transcriptContent').textContent =
        'Transcript will appear here after processing.';
      document.getElementById('summaryContent').className = 'md placeholder';
      document.getElementById('summaryContent').textContent =
        'Structured summary will appear here after processing.';
      document.getElementById('summaryFileLink').innerHTML = '';
      clearErr();
    });
  }

  function joinPath(base, sub) {
    // Build a full path from a base folder + sub-folder name/relative path.
    // Handles both Windows (\) and POSIX (/) separators.
    const sep = base.includes('\\') ? '\\' : '/';
    const b   = base.replace(/[/\\]+$/, '');   // trim trailing slashes
    const s   = sub.trim().replace(/^[/\\]+/, ''); // trim leading slashes
    return s ? b + sep + s : b;
  }

  function autoFillSubPaths() {
    const base = document.getElementById('vault_path').value.trim();
    if (!base) return;
    const audio   = document.getElementById('audio_path').value.trim();
    const summary = document.getElementById('summary_path').value.trim();
    const raw     = document.getElementById('raw_path').value.trim();
    // Only auto-fill if field is currently empty or is a full path we previously set
    if (!audio   || audio.startsWith(base))   document.getElementById('audio_path').value   = joinPath(base, 'Squirrel Notes\\audio');
    if (!summary || summary.startsWith(base)) document.getElementById('summary_path').value = joinPath(base, 'Squirrel Notes\\summaries');
    if (!raw     || raw.startsWith(base))     document.getElementById('raw_path').value     = joinPath(base, 'Squirrel Notes\\raw');
  }

  function browseFolder() {
    fetch('/api/browse-folder')
      .then(r => r.json())
      .then(d => {
        if (d.error) { showErr('Folder picker error: ' + d.error); return; }
        if (!d.path) return; // user cancelled
        document.getElementById('vault_path').value   = d.path;
        document.getElementById('audio_path').value   = d.path + '\\Squirrel Notes\\audio';
        document.getElementById('summary_path').value = d.path + '\\Squirrel Notes\\summaries';
        document.getElementById('raw_path').value     = d.path + '\\Squirrel Notes\\raw';
      })
      .catch(e => showErr('Could not open folder picker: ' + e));
  }

  const _DEFAULT_INSTRUCTIONS = `You are Chip, a highly disciplined senior software engineer squirrel with the attention span of a golden retriever at a tennis ball factory — except every single bit of it is laser-focused on this meeting. You have an almost pathological need to document everything. You have been sent to observe this meeting from a nearby oak tree with a tiny notepad. Your job is to collect every acorn of useful information and bury it in a structured, professional summary that the team can dig up later.

You document meetings covering Jira Issues, Stories and Epics, business requirements, requirement changes, brainstorming sessions, and technical challenges. These notes will be used to develop software solutions, so accuracy and technical detail are critical.

Squirrel Code of Conduct:
- ZERO hallucination. Squirrels of integrity do not bury fake nuts. Do not fabricate speaker names, dates, Jira numbers, table names, or tasks.
- Flag anything unclear with [unclear] or TBD. If you didn't hear it clearly from the tree, say so.
- Be concise but thorough. Every acorn matters.
- Make all section headers bold.

Output Format — follow this structure exactly:

**Topic(s):** [What was this meeting about?]
**Attendees:** [List all speakers/participants identified in the transcript]

---

**Key Discussion Points & Decisions**
1. [First main topic] — [Bullet points with details and decisions]
2. [Next topic] — [Continue as needed]

---

**Next Steps & Timeline**
[Bullet points: what happens next and by when, if mentioned]

---

**Jira Issues, Stories & Epics**
| Jira Number | Title / Subject | Additional Details |
|-------------|-----------------|-------------------|
| ...         | ...             | ...               |

---

**Action Items & Owners**
| Action Item | Owner | Due Date | Additional Details |
|-------------|-------|----------|--------------------|
| TODO: [item] | Christie | [date or TBD] | [notes] |
| ...          | ...   | ...      | ...                |
(Prefix any action item where Christie is the owner with "TODO:")

---

**Points of Contact**
| Name | Role / Department | Contact Context |
|------|-------------------|-----------------|
| ...  | ...               | ...             |

---

**Summary**
[One or two paragraphs wrapping up what was decided and what the overall outcome of the meeting was. Neutral, professional, human-sounding — Chip is thorough but he knows when to wrap it up.]

---

**Technical Summary**
[Summarize all technical items: server names, database names, stored procedures, filenames, function names, APIs, services, etc. Be thorough — this is the good stuff.]

**Tables & Columns**
| Table (camelCase) | Column (camelCase) | Details / FK Relationships |
|-------------------|--------------------|---------------------------|
| ...               | ...                | ...                       |

(All table and column names must be in camelCase — no underscores, no spaces. Note any foreign key relationships in the Details column.)

---

Instructions for Chip the Squirrel:
1. Read the full transcript carefully before writing anything. Every word is an acorn.
2. Identify all unique speakers. If names are missing, use roles (e.g., "Product Manager," "Developer 2").
3. Segment the transcript into discussion topics based on natural transitions and subject changes.
4. Highlight all decisions — look for phrases like "we agreed," "let's go with," "we'll move forward with," "we decided."
5. Extract all Jira numbers, titles, and any history or context discussed around them.
6. Convert every concrete task into an Action Item row. If Christie is the owner, prepend with TODO:.
7. Extract anyone mentioned as a point of contact, product owner, or subject matter expert.
8. In the Technical Summary, capture every technical detail mentioned: servers, databases, tables, columns, stored procedures, code files, APIs, services, and architectures.
9. All table/column names go in the Tables & Columns section in camelCase. Note FK relationships.
10. Close with a crisp, high-level Summary paragraph — the nut buried at the top of the cache.`;

  function resetInstructions() {
    if (confirm('Reset Meeting Instructions to the Chip the Squirrel default? This will overwrite what is currently in the box (but won\'t save until you click Save Settings).')) {
      document.getElementById('meeting_instructions').value = _DEFAULT_INSTRUCTIONS;
    }
  }

  function saveSettings() {
    const devVal = document.getElementById('devSel').value;
    const payload = {
      wmodel:       document.getElementById('wmodel').value,
      omodel:       document.getElementById('omodel').value,
      ollama_base:          document.getElementById('ollama_base').value,
      meeting_instructions: document.getElementById('meeting_instructions').value,
      vault_path:           document.getElementById('vault_path').value,
      audio_path:   document.getElementById('audio_path').value,
      summary_path: document.getElementById('summary_path').value,
      raw_path:     document.getElementById('raw_path').value,
      device_index: devVal !== '' ? parseInt(devVal) : null,
    };
    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(() => {
      const confirm = document.getElementById('saveConfirm');
      confirm.style.display = 'inline';
      setTimeout(() => confirm.style.display = 'none', 2500);
    });
  }

  function loadOllamaModels(currentModel) {
    const badge = document.getElementById('ollamaLoadingBadge');
    badge.textContent = 'loading…';
    badge.style.display = 'inline';
    fetch('/api/ollama-models').then(r => r.json()).then(data => {
      const sel   = document.getElementById('omodel');
      badge.style.display = 'none';
      sel.innerHTML = '';
      if (data.error || !data.models.length) {
        // Keep as a <select> but show a helpful offline message
        const optOffline = document.createElement('option');
        optOffline.value = '';
        optOffline.textContent = '⚠ Ollama offline — run: ollama serve';
        optOffline.disabled = true;
        sel.appendChild(optOffline);
        if (currentModel) {
          const optCurrent = document.createElement('option');
          optCurrent.value = currentModel;
          optCurrent.textContent = currentModel + ' (last used)';
          optCurrent.selected = true;
          sel.appendChild(optCurrent);
        } else {
          optOffline.selected = true;
        }
        return;
      }
      data.models.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === currentModel) opt.selected = true;
        sel.appendChild(opt);
      });
      // If no match found, set first option
      if (!sel.value && data.models.length) sel.value = data.models[0];
    }).catch(() => {
      document.getElementById('ollamaLoadingBadge').textContent = '(offline)';
    });
  }

  function obsidianUrl(filePath) {
    // Build an obsidian:// URI that opens the file directly in Obsidian
    return 'obsidian://open?path=' + encodeURIComponent(filePath);
  }

  // Obsidian logo mark — purple gem shape
  const OB_SVG = '<svg width="15" height="15" viewBox="0 0 24 24" fill="#7c6af7" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M 6 3 L 3 9 L 6 21 L 12 23 L 18 21 L 21 9 L 18 3 Z"/>' +
    '<path d="M 12 3 L 18 3 L 21 9 L 12 23 Z" fill="#a78bfa" opacity=".6"/>' +
    '</svg>';

  // Store file refs by index so onclick never has to embed raw paths in strings
  window._sqFiles = [];
  function _sqIdx(fileObj) {
    window._sqFiles.push(fileObj);
    return window._sqFiles.length - 1;
  }
  function sqPreview(i)  { const f = window._sqFiles[i]; previewFile(f.path, f.name); }
  function sqOpen(i)     { openFile(window._sqFiles[i].path); }
  function sqObsidian(i) { window.location.href = obsidianUrl(window._sqFiles[i].path); }

  function loadRecent() {
    closePreview();
    window._sqFiles = []; // reset index on each load
    fetch('/api/recent').then(r => r.json()).then(data => {
      const el = document.getElementById('recentContent');

      // Group all files by base timestamp name
      const byBase = {};
      const stripExt = f => f.name.replace(/\.(md|txt|wav)$/, '');
      (data.summaries || []).forEach(f => { const k = stripExt(f); byBase[k] = byBase[k]||{}; byBase[k].md  = f; });
      (data.raw       || []).forEach(f => { const k = stripExt(f); byBase[k] = byBase[k]||{}; byBase[k].txt = f; });
      (data.audio     || []).forEach(f => { const k = stripExt(f); byBase[k] = byBase[k]||{}; byBase[k].wav = f; });

      const keys = Object.keys(byBase).sort().reverse();
      if (!keys.length) {
        el.className = 'placeholder';
        el.textContent = 'No recordings yet.';
        return;
      }

      el.className = '';
      let h = '<table class="recent-table"><thead><tr>';
      h += '<th style="text-align:left">Recording</th>';
      h += '<th title="Preview summary">👁</th>';
      h += '<th title="Raw transcript">📝</th>';
      h += '<th title="Open audio">🔊</th>';
      h += '<th title="Open in Obsidian">' + OB_SVG + '</th>';
      h += '</tr></thead><tbody>';

      keys.forEach(key => {
        const { md, txt, wav } = byBase[key];
        h += '<tr><td class="recent-name">' + key.replace(/_/g,' ') + '.md</td>';

        // 👁 Preview markdown
        if (md) {
          const i = _sqIdx(md);
          h += '<td style="text-align:center"><button class="icon-btn" onclick="sqPreview(' + i + ')" title="Preview summary">👁</button></td>';
        } else {
          h += '<td style="text-align:center"><span class="icon-btn disabled">👁</span></td>';
        }

        // 📝 Raw transcript
        if (txt) {
          const i = _sqIdx(txt);
          h += '<td style="text-align:center"><button class="icon-btn" onclick="sqPreview(' + i + ')" title="Raw transcript">📝</button></td>';
        } else {
          h += '<td style="text-align:center"><span class="icon-btn disabled">📝</span></td>';
        }

        // 🔊 Audio
        if (wav) {
          const i = _sqIdx(wav);
          h += '<td style="text-align:center"><button class="icon-btn" onclick="sqOpen(' + i + ')" title="Open audio">🔊</button></td>';
        } else {
          h += '<td style="text-align:center"><span class="icon-btn disabled">🔊</span></td>';
        }

        // Obsidian
        if (md) {
          const i = _sqIdx(md);
          h += '<td style="text-align:center"><button class="icon-btn ob-btn" onclick="sqObsidian(' + i + ')" title="Open in Obsidian">' + OB_SVG + '</button></td>';
        } else {
          h += '<td style="text-align:center"><span class="icon-btn disabled">' + OB_SVG + '</span></td>';
        }

        h += '</tr>';
      });

      h += '</tbody></table>';
      el.innerHTML = h;
    });
  }

  function openFile(path) {
    fetch('/api/open-file?path=' + encodeURIComponent(path)).catch(() => {});
  }

  function previewFile(filePath, fileName) {
    const panel = document.getElementById('previewPanel');
    const body  = document.getElementById('previewBody');
    const title = document.getElementById('previewTitle');
    title.textContent = fileName;
    body.innerHTML = '<span style="color:#484f58;font-size:.8rem">Loading…</span>';
    panel.classList.add('open');
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    fetch('/api/file-content?path=' + encodeURIComponent(filePath))
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          body.innerHTML = '<span style="color:#f85149">' + d.error + '</span>';
          return;
        }
        // .md files get markdown rendering; .txt files get plain pre
        if (fileName.endsWith('.md')) {
          body.innerHTML = marked.parse(d.content);
        } else {
          body.innerHTML = '<pre style="white-space:pre-wrap;font-size:.8rem;color:#c9d1d9">' +
            d.content.replace(/</g,'&lt;') + '</pre>';
        }
      })
      .catch(e => { body.innerHTML = '<span style="color:#f85149">Error: ' + e + '</span>'; });
  }

  function closePreview() {
    document.getElementById('previewPanel').classList.remove('open');
  }

  // ── Status polling ──
  function updateUI(s) {
    const STATUS_LABELS = {
      idle:'Idle', recording:'Recording', processing:'Processing',
      transcribing:'Transcribing', summarizing:'Summarizing',
      done:'Done', error:'Error'
    };
    const busy = ['processing','transcribing','summarizing'].includes(s.status);

    // Timer
    const timer = document.getElementById('timer');
    timer.textContent = fmt(s.duration || 0);
    timer.className = 'timer' + (s.status === 'recording' ? ' live' : '');

    // Status dot + text
    document.getElementById('dot').className =
      'dot' + (s.status === 'recording' ? ' live' : '');
    document.getElementById('statusTxt').textContent =
      STATUS_LABELS[s.status] || s.status;

    // Button
    isRecording = s.status === 'recording';
    const btn = document.getElementById('btnRec');
    btn.disabled  = busy;
    btn.className = 'btn-rec' + (isRecording ? ' live' : '');
    document.getElementById('btnLabel').textContent =
      isRecording ? 'Stop Recording' : 'Start Recording';
    document.getElementById('btnIcon').innerHTML = isRecording
      ? '<rect x="4" y="4" width="16" height="16" rx="2"/>'
      : '<circle cx="12" cy="12" r="8"/>';

    // Progress message
    document.getElementById('progressMsg').textContent = s.message || '';

    // Error
    if (s.status === 'error' && s.error) showErr(s.error);

    // Transcript (populate once it arrives)
    if (s.transcript && lastStatus !== 'done' || s.status === 'done') {
      if (s.transcript) {
        const tc = document.getElementById('transcriptContent');
        tc.className = '';
        const pre = document.createElement('pre');
        pre.className = 'transcript';
        pre.textContent = s.transcript;
        tc.innerHTML = '';
        tc.appendChild(pre);
      }
    }

    // Summary
    if (s.summary) {
      const sc = document.getElementById('summaryContent');
      sc.className = 'md';
      sc.innerHTML = marked.parse(s.summary);
    }

    // File links
    if (s.summary_file || s.raw_file) {
      let links = '';
      if (s.summary_file)
        links += '<div class="file-link">&#128196; Summary: ' + s.summary_file + '</div>';
      if (s.raw_file)
        links += '<div class="file-link" style="margin-top:.35rem">&#128220; Raw transcript: ' + s.raw_file + '</div>';
      document.getElementById('summaryFileLink').innerHTML = links;
    }

    lastStatus = s.status;
  }

  function poll() {
    fetch('/api/status')
      .then(r => r.json())
      .then(updateUI)
      .catch(() => {}); // silent on network error
  }

  // Load devices — auto-select C922 if present
  fetch('/api/devices').then(r => r.json()).then(devs => {
    const sel = document.getElementById('devSel');
    let c922idx = null;
    devs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name;
      sel.appendChild(opt);
      if (c922idx === null && d.name.toLowerCase().includes('c922'))
        c922idx = d.index;
    });
    // Pre-select C922 if found, otherwise leave on auto-detect
    if (c922idx !== null) sel.value = c922idx;
  });

  fetch('/api/settings').then(r => r.json()).then(s => {
    if (s.wmodel)                document.getElementById('wmodel').value                = s.wmodel;
    if (s.ollama_base)           document.getElementById('ollama_base').value           = s.ollama_base;
    if (s.meeting_instructions)  document.getElementById('meeting_instructions').value  = s.meeting_instructions;
    if (s.vault_path)   document.getElementById('vault_path').value   = s.vault_path;
    // Show full paths in sub-folder fields (they store full paths, not just names)
    if (s.audio_path)   document.getElementById('audio_path').value   = s.audio_path;
    if (s.summary_path) document.getElementById('summary_path').value = s.summary_path;
    if (s.raw_path)     document.getElementById('raw_path').value     = s.raw_path;
    // Load Ollama model dropdown after we know the saved model name
    loadOllamaModels(s.omodel || '');
    // Pre-select saved device (done after device list loads)
    if (s.device_index != null) {
      // devices may not be loaded yet — watch for them
      const trySet = setInterval(() => {
        const sel = document.getElementById('devSel');
        if (sel.options.length > 1) {
          sel.value = s.device_index;
          clearInterval(trySet);
        }
      }, 200);
    }
  });

  // Poll every second
  setInterval(poll, 1000);
  poll();

  // Frosted navbar scroll effect
  window.addEventListener('scroll', () => {
    document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 8);
  }, {passive:true});
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing = []
    if not HAS_AUDIO:
        missing.append("sounddevice soundfile")
    if not HAS_WHISPER:
        missing.append("openai-whisper")
    if not HAS_REQUESTS:
        missing.append("requests")

    if missing:
        print("\n⚠  Missing dependencies. Run:")
        print(f"   pip install {' '.join(missing)}\n")

    print("=" * 50)
    print("  🐿️  Squirrel Notes")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
