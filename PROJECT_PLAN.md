# Build Prompt: Universal FFmpeg MCP Server (with optional local UI)

Paste this whole document into Claude Code (or another AI coding agent) as the starting instruction for the project. It describes what to build and why, not how to lay out the repo. Design the folder structure, module boundaries, and file names yourself, whatever fits the tech choices below best.

## 1. What we're building

An MCP server, written in Python, that gives any MCP client (Claude Code, Claude Desktop, Cursor, etc.) full video and image editing power through ffmpeg, plus AI-assisted editing through Whisper and OpenCV/MediaPipe. On top of that, a separate, optional local UI that can be launched independently to watch jobs run and preview results before committing to a final render.

The MCP server:

- Works on macOS, Windows, and Linux with zero manual setup. On first run it checks for a system ffmpeg/ffprobe, and if none is found (or the version is too old), it downloads a static build for the current OS/architecture, verifies its checksum, and caches it locally.
- Exposes ffmpeg's editing surface as a set of typed MCP tools rather than one raw "run ffmpeg command" tool, so the calling model gets guardrails and useful parameters instead of having to write filter graphs by hand.
- Adds three AI layers on top of raw ffmpeg: Whisper for transcription and auto-captioning, MediaPipe/OpenCV for face detection, tracking, and privacy blurring, and a filter-graph builder that turns high-level requests ("cut to vertical and follow the speaker's face") into the correct low-level filter chain.
- Handles long-running jobs (a 4K color grade can take minutes) through an async job pattern instead of blocking the MCP call.

The UI is a thin layer over the same backend, not a second implementation of the editing logic. More on that in phase 6.

## 2. Tech stack decisions and why

- **Language: Python 3.11+, managed with `uv`.** Whisper and MediaPipe are Python-native. Building this in TypeScript would mean shelling out to a Python subprocess for every AI call, which adds latency and a second runtime to install. ffmpeg itself is invoked as a subprocess regardless of host language, so Python costs nothing there.
- **MCP SDK:** official `mcp` Python SDK, stdio transport by default so it drops straight into Claude Desktop / Claude Code config. Leave room for an HTTP/SSE transport later if remote use ever matters.
- **Validation:** pydantic v2 models for every tool's input and output. No raw dicts crossing the tool boundary.
- **Subprocess execution:** always pass ffmpeg arguments as a list, never `shell=True`. Filter strings get complex enough that someone will eventually build one via string concatenation of user input, and that's a shell injection waiting to happen if it's not centralized and guarded.
- **Whisper:** `faster-whisper` (CTranslate2-based), not the original OpenAI Python package. Noticeably faster on CPU, which matters since we can't assume the user has a GPU.
- **Vision:** `mediapipe` for face detection/landmarks, `opencv-python` for frame I/O, blurring, and anything MediaPipe doesn't cover.
- **UI (phase 6):** FastAPI backend for the local server, serving a REST/WebSocket API (WebSocket matters here, polling for job progress is fine for 6a but a scrubbable timeline synced to playback wants push updates). Frontend should be React, since the timeline editor in 6c needs real component state and drag interactions that get painful to hand-roll in plain HTML/JS. No separate database, it reads from the same job store the MCP server writes to.
- **Packaging:** `pyproject.toml`, `uv sync` / `uv run`. No pip/venv/pyenv anywhere in docs or scripts.
- **Formatting/linting:** `ruff` (lint + format), type hints enforced with `mypy`.

## 3. Job model (why not just block)

Some of these operations take a while: a Whisper transcription of a 20-minute video, or a color-graded 4K export. Blocking the MCP call for minutes is bad UX and risks client-side timeouts. Use this pattern instead:

- Every tool that does real work (anything that shells out to ffmpeg or runs a model) returns immediately with a job id and a "queued" status.
- A background worker pool picks up queued jobs, executes them, and updates status (queued, running, done, failed), including a progress percentage parsed from ffmpeg's own progress output (`-progress pipe:1` gives clean key=value pairs, parse `out_time` against the total duration from a probe call).
- A status tool returns current progress for a job id.
- A result tool returns the output file path(s) once done, and errors clearly if called before completion.
- A cancel tool terminates the subprocess and marks the job cancelled.
- Jobs and their output files live in a configured workspace with a retention policy (clean up anything older than a day, for example), not scattered arbitrarily across the filesystem.

Build this job layer before any editing tool. Everything else depends on it, including the UI in phase 6, which is really just a client of this same job store.

## 4. Security and sandboxing

This server touches the filesystem and shells out to a compiled binary, so a few rules are non-negotiable:

- All input/output paths get validated against a configured allowlist of root directories. Reject any path that resolves (after following symlinks) outside those roots. This stops a malicious or confused client from reading or writing outside the intended workspace.
- Never build filter strings by directly interpolating user-supplied text into a filter graph without escaping. ffmpeg filter syntax uses colons, commas, quotes, and brackets as structural characters. The caption and text-overlay tools especially need proper escaping, or a caption containing a colon will silently break the whole graph.
- Enforce a max input file size and max job duration, with a clear error rather than letting a huge file hang the worker pool indefinitely.
- Log every job's resolved ffmpeg command line for auditability, but never log full file contents.
- The UI, once it exists, is the same trust boundary. If it's reachable from the local network rather than just localhost, add auth before exposing it, since it can trigger arbitrary local file reads/writes through the same job tools.

## 5. Coding standards

- Every function that's part of the public tool surface has a docstring and full type hints.
- Every tool's input and output is a pydantic model, defined once and reused everywhere it's needed. No duplicate ad hoc dicts.
- A custom exception hierarchy (binary not found, invalid path, job timeout, unsupported codec, and so on) so the server returns structured error responses instead of raw tracebacks.
- All ffmpeg subprocess calls go through one shared execution path, so timeout handling, progress parsing, and error wrapping live in one place instead of being reimplemented per tool.
- Unit tests cover the pure logic: filter-string builders, path validation, expression generation, schema validation, all runnable without ffmpeg installed.
- Integration tests run against tiny fixture files (a couple seconds of video, a small image) and are the only tests that need a real ffmpeg binary. Keep them fast and mark them separately from unit tests.

## 6. Tool list by phase

Build in this order. Each phase should be a working, tested increment, not a stub.

### Phase 1: Foundation

- Probe media: wraps ffprobe, returns duration, resolution, fps, codecs, bitrate, audio channels, container format.
- Trim: cut a segment, stream-copy when codecs allow (fast path), re-encode otherwise.
- Concat: join clips, normalizing mismatched codecs/resolutions before joining if needed.
- Convert format: transcode between containers/codecs.
- Crop/scale/rotate: geometric transforms, composable in one filter chain.
- Speed ramp: adjust playback speed, video and audio independently, with audio pitch handling chained correctly for larger speed changes.
- List capabilities: resolved ffmpeg version, build info, encoder/filter availability, so the calling model can check before assuming something like hardware encoding is available.
- Job status, job result, cancel job.

### Phase 2: Grading and captions

- Color grade: brightness, contrast, saturation, gamma, temperature.
- Apply LUT: load and validate a `.cube` file, apply it.
- Curves adjustment: preset or custom control points.
- Burn captions: from an SRT or ASS file, with font, size, color, position, and outline styling.
- Text overlay: titles and lower-thirds with timing, position, and simple animation (fade, slide), built from time/position expressions rather than one-off hand-written strings per call.
- Build SRT: a pure helper that turns a list of timed text segments into a valid SRT file, no ffmpeg call involved, used by phase 3.

### Phase 3: Whisper transcription

- Transcribe audio: runs faster-whisper, returns timed segments and word-level timestamps if requested.
- Auto-caption: convenience tool chaining transcription, SRT generation, and caption burning into one job.
- Translate transcript: exposes Whisper's built-in translate-to-English mode; note that other target languages aren't supported by Whisper itself and would need a separate step.

### Phase 4: Vision (MediaPipe/OpenCV)

- Detect faces: samples frames, returns per-timestamp bounding boxes and confidence.
- Track and crop: builds a smoothed crop path following a face across the timeline, and renders it as a reframed output (for example horizontal to vertical). Smooth the path, a crop that snaps frame to frame looks worse than a slightly imperfect static crop.
- Blur faces: detect then apply blur inside a mask that tracks each face region, with an option to exclude a primary subject.
- Scene detection: hard-cut timestamps, useful for auto-editing raw footage into clips.

### Phase 5: Transitions, audio, and full composition

- Add transition: cross-fade or wipe-style transitions between two clips.
- Overlay media: picture-in-picture, watermark, or chroma-keyed composites.
- Mix audio: combine tracks, with optional automatic ducking of background music under a voice track.
- Normalize audio: loudness normalization to a target level.
- Fade audio: fade in/out.
- Render timeline: takes a full structured timeline (clips, captions, overlays, audio tracks, transitions, each with in/out points) and renders it as a single pass. This is the "do everything at once" entry point for anyone driving the server from a script rather than calling tools one at a time.

### Phase 6: Local UI (monitoring, preview, and a basic timeline editor)

A separate, launchable-on-demand local web app that sits on top of the same backend the MCP server uses. Not a second implementation, a second front door. This is the biggest single phase, effectively a small non-linear editor in the browser, so build it in the three sub-stages below rather than all at once.

General direction for all three sub-stages: dark UI, a glassmorphic look (translucent panels, soft blur, subtle borders rather than flat cards), with a single orange accent color (`#E8630A`) reserved for active states, the primary action button, and the playhead/selection highlights. Keep everything else neutral so the accent actually reads as an accent. This should look like a considered personal tool, not a default component-library theme.

**6a. Job monitoring** (build first)

- Live job queue: what's running, queued, done, or failed, with progress bars driven by the same progress data the MCP status tool already reports.
- Ability to cancel a running job from the UI, calling the same cancel tool the MCP server exposes.
- This sub-stage alone should be fully usable before moving on. If the job store updates and this view doesn't reflect it within a second or two, fix that before adding preview.

**6b. Preview** (build second)

- Play input and output clips in-browser: trimmed results, burned captions, the face-tracking crop overlaid on the original source so you can confirm it's actually following the right person before trusting the final render.
- A simple side-by-side or toggle view (before/after) for grading and cropping results, since those are easy to get subtly wrong and hard to judge from a still frame.
- Manual, form-based triggering of any phase 1 to 5 tool, for using the editing power directly without an AI client in the loop. Plain forms are fine here, this doesn't need timeline interaction yet.

**6c. Basic timeline editor** (build third, once 6a and 6b are solid)

- A horizontal timeline with separate tracks for video, captions/text overlays, and audio.
- Drag clips to reorder them on the video track, drag clip edges to trim in/out points, with the change reflected live in the preview player.
- Add a caption or text overlay by dragging its block on the caption track and typing the text; add an audio track by dragging an audio file onto the audio track.
- A scrubbable playhead synced to the preview player, so moving the playhead updates what's shown, and playing back moves the playhead.
- When the user is happy with the arrangement, a single render action that converts the current timeline state into the same structured timeline format the `render_timeline` tool already accepts, and submits it as a job, reusing 6a's monitoring view to track it.
- Keep the first version of this genuinely basic: single video track is fine to start, effects and transitions can be added as simple dropdowns per clip rather than a full keyframe UI. The goal is "arrange and trim visually, then render," not replicating Premiere.
- Choose a frontend approach that can actually support drag-and-drop timeline interaction cleanly (a canvas-based timeline, or a component built with an established drag/resize interaction library) rather than trying to fake dragging with plain HTML elements, since that tends to get janky fast.

All three sub-stages read and write the same job store and file workspace the MCP server tools use. The UI runs as its own process (a separate command from the MCP server), so both can run at once, or the UI can run standalone without an MCP client attached at all.

## 7. MCP server registration example

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

## 8. Definition of done, per phase

A phase isn't done until:

- Every tool in it has a schema, a docstring, and at least one unit test.
- At least one integration test exercises the tool against a fixture file and checks the output's duration, resolution, or codec using the probe tool built in phase 1.
- Lint and type checks pass clean.
- The tool is registered and shows up when an MCP client lists tools.
- For phase 6 specifically: 6a is done when the UI can be started and stopped independently of the MCP server and both observe the same running job at the same time; 6b is done when a rendered output (including a face-tracked crop and burned captions) can actually be played and visually verified in-browser; 6c is done when a timeline built entirely by dragging in the UI produces the exact structured timeline the `render_timeline` tool expects, and renders correctly.

Start with phase 1. Once the binary management, job model, and core tools work end to end on at least two of the three target OSes, move to phase 2. Treat phase 6 as optional relative to phases 1 to 5, meaning the MCP server has to be solid on its own first, but build all three of its sub-stages in order once you get there rather than stopping at monitoring.