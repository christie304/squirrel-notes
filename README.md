# 🐿️ Squirrel Notes

**Privacy-first, local AI notes using Whisper + Ollama + Obsidian.**

> AI meeting notes for people who were *definitely* paying attention.

A lightweight localhost web app that records your meetings, transcribes them with [OpenAI Whisper](https://github.com/openai/whisper) (fully offline), and generates structured meeting summaries using a local [Ollama](https://ollama.com) LLM. Outputs are saved as Markdown files — perfect for an [Obsidian](https://obsidian.md) vault.

## Built With

- Whisper: local transcription
- Ollama: local LLM summaries
- Markdown: clean, portable notes
- Obsidian: knowledge base
- Python: `app.py`
- Localhost: simple UI, no cloud, no API, 100% private

## No Cloud, No Commitment, No Compromise

Most AI note tools require cloud services, API keys, and $$$. Squirrel Notes runs locally so you have full control of your recordings, transcripts, and notes.

Built for developers, tinkerers, and ~~paranoid~~ *privacy-conscious* users.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- [Whisper](https://github.com/openai/whisper) installed
- Microphone access
- [Obsidian](https://obsidian.md) (optional really, but *it's dope!*)



## Contributing

Fork it.  Modify it.  Break it.  Make it better.  Pull requests welcome.


## Support

If this project helps you and you want to support **Squirrel Notes**, you can adopt a puppy from the Humane Society and/or just be a rad human. 


## Features

- **One-click recording** - start/stop from the browser UI
- **Local transcription** - Whisper runs on-device, no API key needed
- **Structured summaries** - meeting notes in a consistent, customisable Markdown template
- **Obsidian integration** - click to open summaries directly in Obsidian
- **Recent files browser** - preview transcripts + summaries inline
- **Configurable** - models, prompts, input device, filepaths are set in the Settings tab

## System Requirements

| Requirement | Notes |
|-------------|-------|
| **Disk Space** | 2-4 GB ¯\\_(ツ)_/¯ (for Whisper + Ollama models) |
| **RAM** | 4+ GB minimum; 8+ GB for larger models; tested with 96 GB |
| **GPU** | NVIDIA GPU with CUDA speeds things up - of course |
| **Python 3.9+** | [python.org](https://www.python.org/downloads/) |
| **Ollama** | [ollama.com](https://ollama.com) — install and pull at least one model (i.e. llama3.2:3b) |
| **Obsidian** | [obsidian.md](https://obsidian.md) — for one-click note opening *(optional)*  |

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Download from https://ollama.com, then:
ollama pull llama3.2
# or any other model you prefer
```

### 2. Clone the repo

```bash
git clone https://github.com/christie304/squirrel-notes.git
cd squirrel-notes
```

### 3. Run the app

**Windows — double-click `start.bat`** (recommended)

The launcher will:
- Verify Python is installed
- Install Python dependencies automatically (first run only)
- Open your browser at `http://localhost:5000`

**Or run manually:**

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.


## First-Time Configuration

Open the **Settings** tab before your first recording:

| Setting | Description |
|---------|-------------|
| **Whisper Model** | Transcription quality vs speed — `base` is a good starting point |
| **Ollama Model** | Dropdown of your installed models (requires Ollama running) |
| **Ollama Base URL** | Ollama server address — default `http://localhost:11434` |
| **Input Device** | Microphone or webcam to record from |
| **Obsidian Vault** | Your Obsidian vault folder — sub-folders are created automatically inside |
| **Meeting Instructions** | System prompt that tells the AI what kind of meeting to expect |

All folders are created automatically. Settings are saved to `config.json` in the app directory.


## How It Works

1. **Record** — Click **Start Recording**. The app captures audio from your selected input device.
2. **Transcribe** — When you click **Stop**, Whisper transcribes the audio locally on your machine.
3. **Summarize** — The transcript is sent to your local Ollama model using your custom Meeting Instructions. The AI answers targeted extraction questions and Python assembles the answers into a structured Markdown document.
4. **Save** — Three files are written into your Obsidian vault:
   - `Squirrel Notes/audio/yyyy_mm_dd_hh_mm.wav` — the raw recording
   - `Squirrel Notes/summaries/yyyy_mm_dd_hh_mm.md` — the structured meeting summary
   - `Squirrel Notes/raw/yyyy_mm_dd_hh_mm.txt` — the full Whisper transcript


## Choosing a Whisper Model

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~75 MB | Fastest | Good for clear audio |
| `base` | ~150 MB | Fast | Recommended starting point |
| `small` | ~500 MB | Moderate | Better for accents/noise |
| `medium` | ~1.5 GB | Slower | High accuracy |
| `large` | ~3 GB | Slowest | Best accuracy |

Models download automatically on first use. GPU (CUDA) is used automatically if available; CPU should be fine for `tiny` through `small`.


## Choosing an Ollama Model

Run `ollama list` to see what you have installed. Some good options:

```bash
ollama pull llama3.2        # Fast, good for summarization
ollama pull mistral         # Strong general-purpose model
ollama pull gemma3:27b      # High quality, needs more RAM
```

Any instruction-following model works. A 7B parameter model is usually sufficient for meeting summaries.


## Meeting Instructions

The **Meeting Instructions** field in Settings is the system prompt sent to your Ollama model before each transcription. The default is tuned for software development team meetings (Jira tickets, database schemas, sprint planning, etc.).

To adapt it for a different meeting type, just edit the text and click **Save Settings**.

## File Structure

```
squirrel-notes/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── start.bat           # Windows launcher
├── config.json         # Your settings (auto-generated; consider adding to .gitignore)
└── README.md
```

Output files go inside your configured Obsidian Vault folder, nested under `Squirrel Notes/`.


## Troubleshooting

**"Ollama not reachable" in model dropdown**
Make sure Ollama is running: open a terminal and run `ollama serve`. Then click the ↻ Refresh button next to the model dropdown.

**Recording produces silence or wrong audio**
Go to Settings → Input Device and select your microphone or webcam explicitly.

**Whisper transcription is slow**
Use a smaller model (`tiny` or `base`). If you have an NVIDIA GPU, install the CUDA version of PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Summary fields show "NONE" or are blank**
The recording may have been too short, or the transcript may lack enough detail. Try a longer recording or switch to a more capable Ollama model.


## Configuration

- `config.json` contains local paths and model names (no credentials) which can be configured on the front-end, in the settings tab.


## License

MIT — use freely, modify as you like.

