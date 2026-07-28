# ffmpeg-mcp

An MCP server that gives any MCP client full video and image editing power through
ffmpeg, plus Whisper transcription and MediaPipe/OpenCV vision tools — and an
optional local web UI for watching jobs, previewing results, and arranging a
timeline.

## Quick start

```bash
uv sync                 # core server
uv sync --all-extras    # plus Whisper, vision, and the UI
uv run ffmpeg-mcp-server
```

Register it with an MCP client:

```json
{
  "mcpServers": {
    "ffmpeg-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/ffmpeg-mcp", "ffmpeg-mcp-server"]
    }
  }
}
```

And, optionally, the local UI — a separate process that can run alongside the
server or entirely on its own:

```bash
uv run ffmpeg-mcp-ui          # http://127.0.0.1:8756
```

## How it is driven

`probe_media` and `list_capabilities` answer immediately. Everything that encodes
frames returns a **job id** straight away; poll `job_status` until it reports
`done`, then call `job_result` for the output path. `cancel_job` stops a queued
or running job.

Outputs go where you name them. Omit `output_path` and the file lands in the
job's own workspace directory, which the retention policy cleans up.

## Tools

**Core** — `probe_media`, `trim`, `concat`, `convert_format`, `transform`,
`speed_ramp`, `list_capabilities`, `job_status`, `job_result`, `cancel_job`,
`list_jobs`

**Colour and text** — `color_grade`, `apply_lut`, `apply_curves`,
`burn_captions`, `text_overlay`, `build_srt`

**Transcription** — `transcribe_audio`, `auto_caption`, `translate_transcript`

**Vision** — `detect_faces`, `track_and_crop`, `blur_faces`, `detect_scenes`

**Composition** — `add_transition`, `overlay_media`, `mix_audio`,
`normalize_audio`, `fade_audio`, `resize_video`, `list_resolution_presets`,
`render_timeline`

### Changing resolution and aspect ratio

`resize_video` handles "make this a Reel" in one call. Named presets cover the
usual targets (`reel`, `tiktok`, `youtube_short`, `story` are all 1080x1920;
`youtube_1080p`, `youtube_4k`, `instagram_square`, `instagram_portrait` do what
they say), or give an explicit width/height, or an `aspect_ratio` like `9:16`.
`list_resolution_presets` enumerates them.

`fit` decides what happens to the picture that no longer fits:

| fit | behaviour |
| --- | --- |
| `cover` (default) | Zooms and crops to fill. Nothing is letterboxed, but the edges are lost. `focus` picks which part survives. |
| `contain` | Fits the whole picture inside and pads with solid bars. |
| `blur` | Fits the whole picture inside over a blurred, zoomed copy of itself — the usual look for turning landscape footage vertical. |
| `stretch` | Distorts the picture to fit exactly. |

For a talking-head video where the subject must stay in frame, prefer
`track_and_crop`, which follows the face rather than cropping to a fixed point.

## Configuration

All settings are environment variables prefixed `FFMPEG_MCP_`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKSPACE` | `~/.ffmpeg-mcp/workspace` | Job outputs, the job store, cached binaries and models. |
| `ALLOWED_ROOTS` | `$HOME` | `os.pathsep`-separated roots that inputs and outputs must sit under. |
| `MAX_INPUT_BYTES` | 16 GiB | Rejects oversized inputs up front. |
| `MAX_JOB_SECONDS` | 10800 | Wall-clock cap per job. |
| `RETENTION_HOURS` | 24 | How long finished jobs and their files are kept. |
| `WORKER_CONCURRENCY` | 2 | Jobs run at once, per process. |
| `FFMPEG_PATH` / `FFPROBE_PATH` | — | Use specific binaries instead of resolving one. |
| `AUTO_DOWNLOAD` | `1` | Allow downloading a static ffmpeg build. |
| `WHISPER_MODEL` | `base` | Default Whisper size. |
| `WHISPER_COMPUTE_TYPE` | `int8` | CTranslate2 compute type. |
| `LOG_LEVEL` | `INFO` | |

## ffmpeg resolution

On first use the server looks for ffmpeg in this order: the configured paths, a
build cached in the workspace, a system ffmpeg on `PATH` that is version 6 or
newer, and finally a static build downloaded for the current OS and architecture.

Checksum policy: builds from BtbN (Linux, Windows) are verified against the
`checksums.sha256` published with the release, and a mismatch is fatal.
evermeet.cx (macOS) publishes only a GPG signature and no digest, so macOS
downloads are pinned on first use — the hash is recorded in
`<workspace>/bin/manifest.json` and any later download of the same URL must
match it. If you would rather not rely on that, install ffmpeg yourself
(`brew install ffmpeg`) and it will be used instead.

The MediaPipe face-detection model (~230 KB) is downloaded on first use of a
vision tool and verified against a pinned SHA-256.

## Security

- Every input and output path is resolved (following symlinks) and checked
  against `ALLOWED_ROOTS`. Anything outside is rejected.
- ffmpeg arguments are always passed as a list. `shell=True` appears nowhere.
- Filter strings are built in one module with both levels of ffmpeg's filter
  escaping applied, so caption text containing `: , ; [ ] ' \` cannot corrupt a
  filter graph. Overlay text is written to a sidecar file and referenced with
  `textfile=` with `expansion=none`, so it never enters the graph at all.
- Every job records its resolved ffmpeg command line for auditing. File contents
  are never logged.
- The UI is the same trust boundary as the MCP server. Bound to loopback it runs
  unauthenticated; bound to any other address it generates and requires a token.
- Uploads are the one place a client-chosen name becomes a path, so the name is
  rebuilt from scratch: the directory component is discarded (`../../etc/passwd`
  becomes `passwd`), the stem is reduced to `[A-Za-z0-9._-]`, and the extension
  must be a media type the tools actually read.

## The local UI

A separate process over the same workspace and job store, so it can run
alongside the MCP server or standalone. Either side can queue a job and either
side can cancel it — cancellation is a cooperative flag the owning worker picks
up, so it works across processes.

- **Jobs** — live queue with progress bars, pushed over a WebSocket, and cancel.
- **Upload** — drag media in from anywhere on the machine, or click to browse.
  Files land in `<workspace>/uploads` and become editable immediately, so you do
  not have to copy them under an allowed root by hand. Filenames are rebuilt
  rather than trusted, the extension must be one the tools handle, and the size
  cap is applied while streaming.
- **Preview** — plays inputs and outputs in-browser with a before/after
  comparison whose two players stay in step.
- **Tools** — a form for every tool, generated from its JSON schema.
- **Timeline** — drag clips to reorder, drag their edges to trim, scrub a
  playhead synced to the preview, then render. The render action produces exactly
  the structure `render_timeline` accepts.

### Rebuilding the UI

The built assets are committed, so running the UI needs no Node toolchain. To
change the frontend:

```bash
cd ui-src
npm install
npm run build      # emits into src/ffmpeg_mcp/ui/static/
npm run dev        # or develop against a running backend on :8756
```

## Development

```bash
uv run pytest                      # everything
uv run pytest -m "not integration" # unit tests only; no ffmpeg needed
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Unit tests cover the pure logic — filter builders and escaping, path validation,
the job store, SRT and LUT parsing, tracking geometry, timeline compilation —
and run without ffmpeg installed. Integration tests generate small fixture clips
on first run and check rendered output by probing it.
