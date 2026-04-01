# Corso YouTube surgery draft annotator

Processes **one YouTube video per spreadsheet row**: download a small-quality copy, sample frames, run **local** vision via **Ollama**, write draft labels to a **copy** of the workbook, then delete the video.

## Requirements (install locally)

- **Python 3.10+**
- **[Ollama](https://ollama.com)** with a vision model, e.g. `ollama pull qwen2.5vl:7b`
- **`yt-dlp`** on PATH (`pip install yt-dlp` or [standalone](https://github.com/yt-dlp/yt-dlp))
- **ffmpeg** on PATH (recommended for stable frame extraction; OpenCV fallback)

## Setup

```bash
cd corso_youtube_annotator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install yt-dlp   # optional if not using standalone binary
```

Copy `.env.example` to `.env` and adjust as needed.

### YouTube cookies (required for most surgery uploads)

Many rows fail with **“Sign in to confirm your age”** or **“Private video”** unless yt-dlp uses your logged-in session.

1. In a normal browser, log into YouTube and confirm you can open the same videos.
2. Put **one** of these in `.env`:
   - `YTDLP_COOKIES_FROM_BROWSER=chrome` (or `safari`, `firefox`) — yt-dlp reads cookies from that browser; close its windows first.
   - `YTDLP_COOKIES=/full/path/to/cookies.txt` — Netscape-format cookies file ([yt-dlp cookie FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)).

Metadata timeouts default to **400s** (`YTDLP_METADATA_TIMEOUT`) for slow networks.

### Start fresh draft workbook

To delete the previous draft and recopy from the source spreadsheet:

`RESET_DRAFT_WORKBOOK=true python annotate_videos.py`

(or set that variable in `.env` once.)

Rows that **cannot** be downloaded still get `auto_notes` + `low_confidence` so the run continues; fix cookies and use `FORCE_REPROCESS=true` to retry.

Paths (defaults in `annotate_videos.py`):

- **Input:** `../docs/Corso Group YouTube Videos (Random Split 17) - Santosh.xlsx` (relative to this folder when cloned inside ROB590), or set `INPUT_XLSX`.
- **Output:** `Corso Group YouTube Videos (Random Split 17) - Santosh_annotated_draft.xlsx` in this project folder (never overwrites the source file).
- **Sheet:** `youtube_videos` by default; override with env `SHEET_NAME` if your workbook differs.

## Run

```bash
source .venv/bin/activate
python annotate_videos.py
```

Quick test (process **one** row and exit):

```bash
MAX_ROWS_PER_RUN=1 python annotate_videos.py
```

Try a **different spreadsheet row** (same sheet, skip rows above *N*):

```bash
START_EXCEL_ROW=50 MAX_ROWS_PER_RUN=1 python annotate_videos.py
```

Re-try rows that already have `auto_review_status` set:

```bash
FORCE_REPROCESS=true MAX_ROWS_PER_RUN=1 python annotate_videos.py
```

**Frame sampling:** By default about **10%** of the video stream’s frames are sent to Ollama (`FRAME_SAMPLE_RATE=0.1`, uses ffprobe’s frame estimate). Set `MAX_FRAMES_PER_VIDEO` to a positive number to cap cost on long videos (default `0` = no cap beyond the percentage).

Background worker with a log file:

```bash
nohup python annotate_videos.py >> annotate_daemon.log 2>&1 &
echo $! > annotate_daemon.pid
```

### If yt-dlp says “Requested format is not available” or “n challenge solving failed”

YouTube often needs a **JavaScript runtime** for yt-dlp’s challenge solver. Install dependencies and verify:

```bash
pip install -U yt-dlp yt-dlp-ejs
# Node.js should be on PATH (e.g. Homebrew: /opt/homebrew/bin/node)
python -m yt_dlp -v -F "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | grep "JS Challenge"
```

You want to see **node** listed as available, not `node (unavailable)`. Then re-run the annotator from the **same terminal** (so `PATH` matches).

As an alternative JS runtime for yt-dlp, try **`brew install deno`** and run `yt-dlp -v -F` again (some setups only pick up Deno).

By default the script passes **`youtube:player_client=web,default`** (see `YTDLP_EXTRACTOR_ARGS`) to reduce “No video formats” on non–age-gated videos.

Edit the **config block** at the top of `annotate_videos.py` for:

- `INPUT_XLSX`, `OUTPUT_XLSX`, `SHEET_NAME`
- `YOUTUBE_ID_COLUMN` / `YOUTUBE_LINK_COLUMN` (auto-detect if `None`)
- `FRAME_INTERVAL_SEC`, `FRAME_SAMPLE_RATE`, `MAX_FRAMES_PER_VIDEO`, `VISION_IMAGE_MAX_EDGE`
- `OLLAMA_HOST`, `LOCAL_MODEL_NAME`, `FALLBACK_MODELS`
- `DRY_RUN`, `DRY_RUN_ROWS`, `DEBUG_KEEP_VIDEOS`, `FORCE_REPROCESS`

## Switching models

In `annotate_videos.py`, set `LOCAL_MODEL_NAME` to your pulled model, for example:

- `qwen2.5vl:7b` (default)
- `openbmb/minicpm-v4.5` (if pulled as that name in Ollama)
- `llava:7b`

If the primary model errors, the script retries with `FALLBACK_MODELS` in order.

## Safety

- The **original** Excel file is never written; only the `_annotated_draft` copy is updated.
- After each row, the draft workbook is **saved** so a crash keeps completed rows.
- Videos and extracted frames live under `TEMP_DIR` and are removed after each row (unless `DEBUG_KEEP_VIDEOS=true`).

## Troubleshooting

- **`yt-dlp` not found**: install and ensure it is on `PATH`.
- **Ollama connection**: start Ollama; check `OLLAMA_HOST` (default `http://localhost:11434`).
- **Vision model too slow**: reduce `MAX_FRAMES_PER_VIDEO` or increase `FRAME_INTERVAL_SEC`.
- **`GGML_ASSERT(a->ne[2] * 4 == b->ne[0])` / HTTP 500 from Ollama vision**: frames are preprocessed before chat (default max edge 768px, pad to multiples of 28). Lower `VISION_IMAGE_MAX_EDGE` (e.g. `512`) or `FRAMES_PER_OLLAMA_BATCH=1`; upgrade Ollama and re-pull the model.
