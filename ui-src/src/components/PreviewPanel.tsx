/**
 * Phase 6b: playback and before/after comparison.
 *
 * Grading and cropping are easy to get subtly wrong and hard to judge from a
 * still, so the comparison view plays the source and the result side by side
 * with their playback positions kept in step.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { listFiles, mediaUrl, type Job, type MediaFile } from "../api";
import { basename, formatBytes } from "../format";

interface Props {
  jobs: Job[];
  selectedJob: Job | null;
}

type Mode = "single" | "compare";

function sourceOf(job: Job | null): string | null {
  if (!job) return null;
  const params = job.params as Record<string, unknown>;
  for (const key of ["input_path", "first_path", "voice_path"]) {
    const value = params[key];
    if (typeof value === "string") return value;
  }
  const timeline = params.timeline as { clips?: { source?: string }[] } | undefined;
  return timeline?.clips?.[0]?.source ?? null;
}

function kindOf(path: string): MediaFile["kind"] {
  const suffix = path.toLowerCase().split(".").pop() ?? "";
  if (["mp4", "mov", "mkv", "webm", "avi", "m4v"].includes(suffix)) return "video";
  if (["mp3", "wav", "m4a", "flac", "opus", "aac", "ogg"].includes(suffix)) return "audio";
  if (["png", "jpg", "jpeg", "webp", "gif"].includes(suffix)) return "image";
  return "other";
}

function Player({ path, refObject }: { path: string; refObject?: React.Ref<HTMLVideoElement> }) {
  const kind = kindOf(path);
  if (kind === "image") return <img className="preview" src={mediaUrl(path)} alt={basename(path)} />;
  if (kind === "audio") {
    return <audio controls style={{ width: "100%" }} src={mediaUrl(path)} />;
  }
  return <video ref={refObject} controls preload="metadata" src={mediaUrl(path)} />;
}

export default function PreviewPanel({ jobs, selectedJob }: Props) {
  const [files, setFiles] = useState<MediaFile[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("single");
  const [error, setError] = useState<string | null>(null);
  const [linked, setLinked] = useState(true);

  const beforeRef = useRef<HTMLVideoElement>(null);
  const afterRef = useRef<HTMLVideoElement>(null);

  const refresh = async () => {
    try {
      setFiles(await listFiles());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void refresh();
    // Finished jobs produce new files, so refresh when the set of them changes.
  }, [jobs.filter((job) => job.status === "done").length]);

  const jobOutput = selectedJob?.outputs[0] ?? null;
  const active = chosen ?? jobOutput ?? files[0]?.path ?? null;
  const source = useMemo(() => sourceOf(selectedJob), [selectedJob]);
  const canCompare = Boolean(source && jobOutput && source !== jobOutput);

  // Keep the two players in step so a difference is genuinely a difference in
  // the picture rather than in the playback position.
  useEffect(() => {
    if (mode !== "compare" || !linked) return;
    const before = beforeRef.current;
    const after = afterRef.current;
    if (!before || !after) return;

    const sync = (from: HTMLVideoElement, to: HTMLVideoElement) => () => {
      if (Math.abs(to.currentTime - from.currentTime) > 0.15) to.currentTime = from.currentTime;
    };
    const play = (to: HTMLVideoElement) => () => void to.play().catch(() => undefined);
    const pause = (to: HTMLVideoElement) => () => to.pause();

    const handlers: [HTMLVideoElement, string, EventListener][] = [
      [before, "seeked", sync(before, after)],
      [before, "timeupdate", sync(before, after)],
      [before, "play", play(after)],
      [before, "pause", pause(after)],
    ];
    handlers.forEach(([el, event, handler]) => el.addEventListener(event, handler));
    return () => handlers.forEach(([el, event, handler]) => el.removeEventListener(event, handler));
  }, [mode, linked, active, source]);

  return (
    <div className="grid-side">
      <div className="panel">
        <h2 className="panel-title">
          Files
          <button className="btn small" onClick={() => void refresh()}>
            Refresh
          </button>
        </h2>
        {error && <div className="error-banner">{error}</div>}
        {files.length === 0 ? (
          <div className="empty">No media in the workspace yet.</div>
        ) : (
          <div className="file-list">
            {files.map((file) => (
              <div
                key={file.path}
                className="file"
                data-active={file.path === active}
                onClick={() => setChosen(file.path)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => event.key === "Enter" && setChosen(file.path)}
                title={file.path}
              >
                <span className="kind">{file.kind}</span>
                <span className="name">{file.name}</span>
                <span className="spacer" />
                <span className="kind">{formatBytes(file.size_bytes)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <h2 className="panel-title">
          Preview
          <span className="row">
            <button
              className="btn small"
              data-active={mode === "single"}
              onClick={() => setMode("single")}
              style={{ opacity: mode === "single" ? 1 : 0.6 }}
            >
              Single
            </button>
            <button
              className="btn small"
              disabled={!canCompare}
              onClick={() => setMode("compare")}
              style={{ opacity: mode === "compare" ? 1 : 0.6 }}
              title={canCompare ? "" : "Select a finished job to compare against its source"}
            >
              Before / after
            </button>
          </span>
        </h2>

        {!active ? (
          <div className="empty">Choose a file, or select a finished job.</div>
        ) : mode === "compare" && canCompare && source && jobOutput ? (
          <>
            <div className="compare">
              <figure>
                <Player path={source} refObject={beforeRef} />
                <figcaption>Before — {basename(source)}</figcaption>
              </figure>
              <figure>
                <Player path={jobOutput} refObject={afterRef} />
                <figcaption>After — {basename(jobOutput)}</figcaption>
              </figure>
            </div>
            <label className="row" style={{ marginTop: 12 }}>
              <input
                type="checkbox"
                checked={linked}
                onChange={(event) => setLinked(event.target.checked)}
              />
              <span className="hint">Keep playback in step</span>
            </label>
          </>
        ) : (
          <>
            <Player path={active} />
            <div className="job-meta" style={{ marginTop: 10 }}>
              <span className="mono">{active}</span>
            </div>
          </>
        )}

        {selectedJob?.result && Object.keys(selectedJob.result).length > 0 && (
          <details className="raw">
            <summary>Job result</summary>
            <pre className="command">{JSON.stringify(selectedJob.result, null, 2)}</pre>
          </details>
        )}
        {selectedJob?.command && (
          <details className="raw">
            <summary>ffmpeg command</summary>
            <pre className="command">{selectedJob.command}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
