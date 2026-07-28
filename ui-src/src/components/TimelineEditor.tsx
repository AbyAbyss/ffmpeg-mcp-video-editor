/**
 * Phase 6c: a basic non-linear timeline editor.
 *
 * Clips are dragged to reorder and their edges dragged to trim, using
 * interact.js for the pointer handling rather than hand-rolled mouse maths.
 * The playhead is scrubbable and stays in step with the preview player.
 *
 * The render action converts this state into exactly the structure the
 * `render_timeline` tool accepts and submits it as a job, which then shows up
 * in the same monitoring view as everything else.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import interact from "interactjs";
import { callTool, listFiles, mediaUrl, type MediaFile } from "../api";
import { formatDuration } from "../format";

const TRANSITIONS = [
  "none",
  "fade",
  "fadeblack",
  "dissolve",
  "wipeleft",
  "wiperight",
  "wipeup",
  "wipedown",
  "slideleft",
  "slideright",
  "circleopen",
  "circleclose",
];

const PRESETS: Record<string, [number, number]> = {
  "1920x1080 (YouTube)": [1920, 1080],
  "1280x720": [1280, 720],
  "1080x1920 (Reel / Shorts)": [1080, 1920],
  "1080x1080 (Square)": [1080, 1080],
};

interface Clip {
  id: string;
  source: string;
  name: string;
  /** Source in/out points, in seconds. */
  inPoint: number;
  outPoint: number;
  /** Full length of the underlying media. */
  sourceDuration: number;
  speed: number;
  transition: string;
  transitionDuration: number;
  mute: boolean;
}

interface TextItem {
  id: string;
  text: string;
  start: number;
  end: number;
  position: string;
  fontSize: number;
  fade: number;
}

interface AudioItem {
  id: string;
  source: string;
  name: string;
  start: number;
  gainDb: number;
  duck: boolean;
}

let counter = 0;
const nextId = (prefix: string) => `${prefix}-${(counter += 1)}`;

/** Seconds of timeline per pixel is derived from the track width and total length. */
function useTrackWidth(ref: React.RefObject<HTMLElement>) {
  const [width, setWidth] = useState(900);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    setWidth(element.clientWidth);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

export default function TimelineEditor({ onQueued }: { onQueued: () => void }) {
  const [files, setFiles] = useState<MediaFile[]>([]);
  const [clips, setClips] = useState<Clip[]>([]);
  const [texts, setTexts] = useState<TextItem[]>([]);
  const [audio, setAudio] = useState<AudioItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [playhead, setPlayhead] = useState(0);
  const [size, setSize] = useState<[number, number]>([1280, 720]);
  const [fps, setFps] = useState(30);
  const [normalize, setNormalize] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const videoTrackRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<HTMLVideoElement>(null);
  const trackWidth = useTrackWidth(videoTrackRef);

  const clipDuration = useCallback(
    (clip: Clip) => (clip.outPoint - clip.inPoint) / clip.speed,
    [],
  );

  // Clip positions are derived from order and length, with transitions pulling
  // the following clip back over the previous one.
  const layout = useMemo(() => {
    let cursor = 0;
    const placed = clips.map((clip, index) => {
      const previous = clips[index - 1];
      if (previous && previous.transition !== "none") {
        cursor = Math.max(0, cursor - previous.transitionDuration);
      }
      const start = cursor;
      const length = clipDuration(clip);
      cursor = start + length;
      return { clip, start, length };
    });
    return { placed, total: cursor };
  }, [clips, clipDuration]);

  const total = Math.max(layout.total, 10);
  const pxPerSecond = trackWidth / total;
  const toX = (seconds: number) => seconds * pxPerSecond;
  const toSeconds = (x: number) => x / pxPerSecond;

  useEffect(() => {
    void (async () => {
      try {
        setFiles(await listFiles());
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, []);

  const addClip = async (file: MediaFile) => {
    try {
      const info = await callTool<{ duration: number | null }>("probe_media", {
        input_path: file.path,
      });
      const duration = info.duration ?? 5;
      setClips((current) => [
        ...current,
        {
          id: nextId("clip"),
          source: file.path,
          name: file.name,
          inPoint: 0,
          outPoint: duration,
          sourceDuration: duration,
          speed: 1,
          transition: "none",
          transitionDuration: 0.5,
          mute: false,
        },
      ]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const addAudio = async (file: MediaFile) => {
    setAudio((current) => [
      ...current,
      {
        id: nextId("audio"),
        source: file.path,
        name: file.name,
        start: 0,
        gainDb: -8,
        duck: true,
      },
    ]);
  };

  const addText = () => {
    setTexts((current) => [
      ...current,
      {
        id: nextId("text"),
        text: "New title",
        start: playhead,
        end: Math.min(total, playhead + 3),
        position: "bottom-center",
        fontSize: 48,
        fade: 0.3,
      },
    ]);
  };

  // -- interact.js wiring ------------------------------------------------ #

  useEffect(() => {
    const dragged = interact(".clip.video")
      .draggable({
        listeners: {
          start(event) {
            event.target.dataset.dx = "0";
          },
          move(event) {
            const dx = Number(event.target.dataset.dx ?? 0) + event.dx;
            event.target.dataset.dx = String(dx);
            event.target.style.transform = `translateX(${dx}px)`;
          },
          end(event) {
            const dx = Number(event.target.dataset.dx ?? 0);
            event.target.style.transform = "";
            event.target.dataset.dx = "0";
            const id = event.target.dataset.id;
            if (!id || Math.abs(dx) < 4) return;
            // Reorder by where the clip's centre landed.
            setClips((current) => {
              const index = current.findIndex((clip) => clip.id === id);
              if (index === -1) return current;
              const moved = toSeconds(dx);
              const positions = current.map((_, i) => i);
              let target = index;
              let travelled = 0;
              if (moved > 0) {
                for (let i = index + 1; i < current.length; i += 1) {
                  travelled += clipDuration(current[i]);
                  if (travelled <= moved) target = i;
                }
              } else {
                for (let i = index - 1; i >= 0; i -= 1) {
                  travelled -= clipDuration(current[i]);
                  if (travelled >= moved) target = i;
                }
              }
              if (target === index) return current;
              const next = [...current];
              const [clip] = next.splice(index, 1);
              next.splice(target, 0, clip);
              return positions.length ? next : current;
            });
          },
        },
      })
      .resizable({
        edges: { left: ".handle.left", right: ".handle.right" },
        listeners: {
          move(event) {
            const id = event.target.dataset.id;
            if (!id) return;
            const deltaLeft = toSeconds(event.deltaRect?.left ?? 0);
            const deltaRight = toSeconds(event.deltaRect?.right ?? 0);
            setClips((current) =>
              current.map((clip) => {
                if (clip.id !== id) return clip;
                // Trimming moves the source in/out points, scaled by speed so a
                // sped-up clip still trims the right amount of source.
                let inPoint = clip.inPoint + deltaLeft * clip.speed;
                let outPoint = clip.outPoint + deltaRight * clip.speed;
                inPoint = Math.max(0, Math.min(inPoint, clip.outPoint - 0.1));
                outPoint = Math.min(
                  clip.sourceDuration,
                  Math.max(outPoint, inPoint + 0.1),
                );
                return { ...clip, inPoint, outPoint };
              }),
            );
          },
        },
      });

    return () => {
      dragged.unset();
    };
  }, [clipDuration, pxPerSecond]);

  // -- playhead ---------------------------------------------------------- #

  const scrubTo = (clientX: number) => {
    const element = videoTrackRef.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const seconds = Math.max(0, Math.min(total, toSeconds(clientX - rect.left)));
    setPlayhead(seconds);
    const player = playerRef.current;
    const active = layout.placed.find(
      (entry) => seconds >= entry.start && seconds < entry.start + entry.length,
    );
    if (player && active) {
      const into = (seconds - active.start) * active.clip.speed + active.clip.inPoint;
      if (player.dataset.source !== active.clip.source) {
        player.dataset.source = active.clip.source;
        player.src = mediaUrl(active.clip.source);
      }
      player.currentTime = into;
    }
  };

  const onTrackPointerDown = (event: React.PointerEvent) => {
    if ((event.target as HTMLElement).closest(".clip")) return;
    scrubTo(event.clientX);
  };

  // Playing moves the playhead, so the two stay in step both ways.
  useEffect(() => {
    const player = playerRef.current;
    if (!player) return;
    const onTime = () => {
      const active = layout.placed.find(
        (entry) => entry.clip.source === player.dataset.source,
      );
      if (!active) return;
      const offset = (player.currentTime - active.clip.inPoint) / active.clip.speed;
      setPlayhead(Math.max(0, Math.min(total, active.start + offset)));
    };
    player.addEventListener("timeupdate", onTime);
    return () => player.removeEventListener("timeupdate", onTime);
  }, [layout, total]);

  // -- render ------------------------------------------------------------ #

  /** Convert the editor's state into the exact `render_timeline` structure. */
  const buildTimeline = () => ({
    width: size[0],
    height: size[1],
    fps,
    clips: clips.map((clip, index) => ({
      source: clip.source,
      in_point: Number(clip.inPoint.toFixed(3)),
      out_point: Number(clip.outPoint.toFixed(3)),
      speed: clip.speed,
      mute: clip.mute,
      ...(clip.transition !== "none" && index < clips.length - 1
        ? {
            transition_to_next: {
              type: clip.transition,
              duration: clip.transitionDuration,
            },
          }
        : {}),
    })),
    text_overlays: texts.map((text) => ({
      text: text.text,
      start: Number(text.start.toFixed(3)),
      end: Number(text.end.toFixed(3)),
      position: text.position,
      font_size: text.fontSize,
      fade: text.fade,
    })),
    audio_tracks: audio.map((track) => ({
      source: track.source,
      start: Number(track.start.toFixed(3)),
      gain_db: track.gainDb,
      duck_under_voice: track.duck,
    })),
    normalize_audio: normalize,
  });

  const render = async () => {
    if (clips.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await callTool("render_timeline", { timeline: buildTimeline() });
      onQueued();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const selectedClip = clips.find((clip) => clip.id === selected) ?? null;
  const selectedText = texts.find((text) => text.id === selected) ?? null;
  const selectedAudio = audio.find((track) => track.id === selected) ?? null;

  const ticks = useMemo(() => {
    const step = total <= 15 ? 1 : total <= 60 ? 5 : total <= 300 ? 30 : 60;
    const marks: number[] = [];
    for (let t = 0; t <= total; t += step) marks.push(t);
    return marks;
  }, [total]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="grid-side">
        <div className="panel">
          <h2 className="panel-title">Media</h2>
          <div className="file-list">
            {files
              .filter((file) => file.kind === "video" || file.kind === "audio")
              .map((file) => (
                <div key={file.path} className="file" title={file.path}>
                  <span className="kind">{file.kind}</span>
                  <span className="name">{file.name}</span>
                  <span className="spacer" />
                  {file.kind === "video" ? (
                    <button className="btn small" onClick={() => void addClip(file)}>
                      + Video
                    </button>
                  ) : (
                    <button className="btn small" onClick={() => void addAudio(file)}>
                      + Audio
                    </button>
                  )}
                </div>
              ))}
            {files.length === 0 && <div className="empty">No media in the workspace.</div>}
          </div>
        </div>

        <div className="panel">
          <h2 className="panel-title">Preview</h2>
          <video ref={playerRef} controls preload="metadata" />
          <div className="job-meta" style={{ marginTop: 8 }}>
            <span>
              Playhead {formatDuration(playhead)} / {formatDuration(layout.total)}
            </span>
          </div>
        </div>
      </div>

      <div className="timeline-wrap">
        <div className="toolbar">
          <select
            value={`${size[0]}x${size[1]}`}
            onChange={(event) => {
              const entry = Object.entries(PRESETS).find(
                ([, [w, h]]) => `${w}x${h}` === event.target.value,
              );
              if (entry) setSize(entry[1]);
            }}
            style={{ width: "auto" }}
          >
            {Object.entries(PRESETS).map(([label, [w, h]]) => (
              <option key={label} value={`${w}x${h}`}>
                {label}
              </option>
            ))}
          </select>
          <label className="row">
            <span className="hint">fps</span>
            <input
              type="number"
              min={1}
              max={120}
              value={fps}
              onChange={(event) => setFps(Number(event.target.value))}
              style={{ width: 70 }}
            />
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={normalize}
              onChange={(event) => setNormalize(event.target.checked)}
            />
            <span className="hint">Normalise audio</span>
          </label>
          <button className="btn" onClick={addText}>
            + Caption
          </button>
          <span className="spacer" />
          <span className="hint">
            {clips.length} clip{clips.length === 1 ? "" : "s"} ·{" "}
            {formatDuration(layout.total)}
          </span>
          <button
            className="btn primary"
            onClick={() => void render()}
            disabled={busy || clips.length === 0}
          >
            {busy ? "Submitting…" : "Render timeline"}
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div className="ruler" onPointerDown={(event) => scrubTo(event.clientX)}>
          {ticks.map((tick) => (
            <span key={tick} style={{ left: `${toX(tick)}px` }}>
              {formatDuration(tick)}
            </span>
          ))}
        </div>

        <div className="timeline-body">
          <div className="scrub-layer">
            <div className="playhead" style={{ left: `${toX(playhead)}px` }} />
          </div>

          <div className="track-row">
            <div className="track-label">Video</div>
            <div className="track" ref={videoTrackRef} onPointerDown={onTrackPointerDown}>
              {layout.placed.map(({ clip, start, length }) => (
                <div
                  key={clip.id}
                  className="clip video"
                  data-id={clip.id}
                  data-selected={selected === clip.id}
                  style={{ left: `${toX(start)}px`, width: `${Math.max(24, toX(length))}px` }}
                  onPointerDown={() => setSelected(clip.id)}
                >
                  <span className="handle left" />
                  {clip.name}
                  <span className="sub">
                    {formatDuration(length)}
                    {clip.speed !== 1 ? ` · ${clip.speed}x` : ""}
                    {clip.transition !== "none" ? ` · ${clip.transition}` : ""}
                  </span>
                  <span className="handle right" />
                </div>
              ))}
              {clips.length === 0 && (
                <div className="empty" style={{ padding: 14 }}>
                  Add a clip from the media list.
                </div>
              )}
            </div>
          </div>

          <div className="track-row">
            <div className="track-label">Captions</div>
            <div className="track" onPointerDown={onTrackPointerDown}>
              {texts.map((text) => (
                <div
                  key={text.id}
                  className="clip caption"
                  data-id={text.id}
                  data-selected={selected === text.id}
                  style={{
                    left: `${toX(text.start)}px`,
                    width: `${Math.max(24, toX(text.end - text.start))}px`,
                  }}
                  onPointerDown={() => setSelected(text.id)}
                >
                  {text.text}
                  <span className="sub">{formatDuration(text.end - text.start)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="track-row">
            <div className="track-label">Audio</div>
            <div className="track" onPointerDown={onTrackPointerDown}>
              {audio.map((track) => (
                <div
                  key={track.id}
                  className="clip audio"
                  data-id={track.id}
                  data-selected={selected === track.id}
                  style={{
                    left: `${toX(track.start)}px`,
                    width: `${Math.max(40, toX(Math.max(2, total - track.start)))}px`,
                  }}
                  onPointerDown={() => setSelected(track.id)}
                >
                  {track.name}
                  <span className="sub">
                    {track.gainDb} dB{track.duck ? " · ducked" : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {(selectedClip || selectedText || selectedAudio) && (
        <div className="panel">
          <h2 className="panel-title">
            Selected
            <button
              className="btn small danger"
              onClick={() => {
                setClips((c) => c.filter((clip) => clip.id !== selected));
                setTexts((t) => t.filter((text) => text.id !== selected));
                setAudio((a) => a.filter((track) => track.id !== selected));
                setSelected(null);
              }}
            >
              Remove
            </button>
          </h2>

          {selectedClip && (
            <div className="inspector">
              <div className="field">
                <label>In point (s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={selectedClip.inPoint.toFixed(2)}
                  onChange={(event) =>
                    setClips((current) =>
                      current.map((clip) =>
                        clip.id === selectedClip.id
                          ? {
                              ...clip,
                              inPoint: Math.max(
                                0,
                                Math.min(Number(event.target.value), clip.outPoint - 0.1),
                              ),
                            }
                          : clip,
                      ),
                    )
                  }
                />
              </div>
              <div className="field">
                <label>Out point (s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={selectedClip.outPoint.toFixed(2)}
                  onChange={(event) =>
                    setClips((current) =>
                      current.map((clip) =>
                        clip.id === selectedClip.id
                          ? {
                              ...clip,
                              outPoint: Math.min(
                                clip.sourceDuration,
                                Math.max(Number(event.target.value), clip.inPoint + 0.1),
                              ),
                            }
                          : clip,
                      ),
                    )
                  }
                />
              </div>
              <div className="field">
                <label>Speed</label>
                <input
                  type="number"
                  step="0.25"
                  min="0.1"
                  max="10"
                  value={selectedClip.speed}
                  onChange={(event) =>
                    setClips((current) =>
                      current.map((clip) =>
                        clip.id === selectedClip.id
                          ? { ...clip, speed: Math.max(0.1, Number(event.target.value)) }
                          : clip,
                      ),
                    )
                  }
                />
              </div>
              <div className="field">
                <label>Transition to next</label>
                <select
                  value={selectedClip.transition}
                  onChange={(event) =>
                    setClips((current) =>
                      current.map((clip) =>
                        clip.id === selectedClip.id
                          ? { ...clip, transition: event.target.value }
                          : clip,
                      ),
                    )
                  }
                >
                  {TRANSITIONS.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
              {selectedClip.transition !== "none" && (
                <div className="field">
                  <label>Transition length (s)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={selectedClip.transitionDuration}
                    onChange={(event) =>
                      setClips((current) =>
                        current.map((clip) =>
                          clip.id === selectedClip.id
                            ? {
                                ...clip,
                                transitionDuration: Math.max(0.1, Number(event.target.value)),
                              }
                            : clip,
                        ),
                      )
                    }
                  />
                </div>
              )}
              <div className="field">
                <label className="row">
                  <input
                    type="checkbox"
                    checked={selectedClip.mute}
                    onChange={(event) =>
                      setClips((current) =>
                        current.map((clip) =>
                          clip.id === selectedClip.id
                            ? { ...clip, mute: event.target.checked }
                            : clip,
                        ),
                      )
                    }
                  />
                  <span>Mute this clip</span>
                </label>
              </div>
            </div>
          )}

          {selectedText && (
            <div className="inspector">
              <div className="field" style={{ gridColumn: "1 / -1" }}>
                <label>Text</label>
                <input
                  type="text"
                  value={selectedText.text}
                  onChange={(event) =>
                    setTexts((current) =>
                      current.map((text) =>
                        text.id === selectedText.id ? { ...text, text: event.target.value } : text,
                      ),
                    )
                  }
                />
              </div>
              {(
                [
                  ["start", "Start (s)"],
                  ["end", "End (s)"],
                  ["fontSize", "Font size"],
                  ["fade", "Fade (s)"],
                ] as const
              ).map(([key, label]) => (
                <div className="field" key={key}>
                  <label>{label}</label>
                  <input
                    type="number"
                    step={key === "fontSize" ? 1 : 0.1}
                    value={selectedText[key]}
                    onChange={(event) =>
                      setTexts((current) =>
                        current.map((text) =>
                          text.id === selectedText.id
                            ? { ...text, [key]: Number(event.target.value) }
                            : text,
                        ),
                      )
                    }
                  />
                </div>
              ))}
              <div className="field">
                <label>Position</label>
                <select
                  value={selectedText.position}
                  onChange={(event) =>
                    setTexts((current) =>
                      current.map((text) =>
                        text.id === selectedText.id
                          ? { ...text, position: event.target.value }
                          : text,
                      ),
                    )
                  }
                >
                  {[
                    "top-left",
                    "top-center",
                    "top-right",
                    "center",
                    "bottom-left",
                    "bottom-center",
                    "bottom-right",
                  ].map((position) => (
                    <option key={position} value={position}>
                      {position}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {selectedAudio && (
            <div className="inspector">
              <div className="field">
                <label>Start (s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={selectedAudio.start}
                  onChange={(event) =>
                    setAudio((current) =>
                      current.map((track) =>
                        track.id === selectedAudio.id
                          ? { ...track, start: Math.max(0, Number(event.target.value)) }
                          : track,
                      ),
                    )
                  }
                />
              </div>
              <div className="field">
                <label>Gain (dB)</label>
                <input
                  type="number"
                  step="1"
                  value={selectedAudio.gainDb}
                  onChange={(event) =>
                    setAudio((current) =>
                      current.map((track) =>
                        track.id === selectedAudio.id
                          ? { ...track, gainDb: Number(event.target.value) }
                          : track,
                      ),
                    )
                  }
                />
              </div>
              <div className="field">
                <label className="row">
                  <input
                    type="checkbox"
                    checked={selectedAudio.duck}
                    onChange={(event) =>
                      setAudio((current) =>
                        current.map((track) =>
                          track.id === selectedAudio.id
                            ? { ...track, duck: event.target.checked }
                            : track,
                        ),
                      )
                    }
                  />
                  <span>Duck under the clips' audio</span>
                </label>
              </div>
            </div>
          )}
        </div>
      )}

      <details className="raw">
        <summary>Structured timeline this will render</summary>
        <pre className="command">{JSON.stringify(buildTimeline(), null, 2)}</pre>
      </details>
    </div>
  );
}
