/**
 * Phase 6a: the live job queue.
 *
 * Job state arrives over the WebSocket rather than being polled from the
 * browser, so a job started by an attached MCP client shows up here just as
 * quickly as one started from these panels.
 */

import { useMemo, useState } from "react";
import { cancelJob, type Job } from "../api";
import { basename, formatElapsed } from "../format";

const STATUS_ORDER = ["running", "queued", "done", "failed", "cancelled"] as const;

interface Props {
  jobs: Job[];
  counts: Record<string, number>;
  /** Only worth showing when the list spans more than one project. */
  showProject: boolean;
  selectedId: string | null;
  onSelect: (job: Job) => void;
}

export default function JobsPanel({ jobs, counts, showProject, selectedId, onSelect }: Props) {
  const [filter, setFilter] = useState<string>("all");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const visible = useMemo(() => {
    const list = filter === "all" ? jobs : jobs.filter((job) => job.status === filter);
    // Active work first, then most recent, so the interesting rows stay on top.
    const rank = (job: Job) =>
      job.status === "running" ? 0 : job.status === "queued" ? 1 : 2;
    return [...list].sort((a, b) => rank(a) - rank(b) || b.created_at - a.created_at);
  }, [jobs, filter]);

  const onCancel = async (job: Job, event: React.MouseEvent) => {
    event.stopPropagation();
    setBusy(job.job_id);
    setError(null);
    try {
      await cancelJob(job.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="counts">
        <button
          className="count"
          data-kind="all"
          onClick={() => setFilter("all")}
          style={{ cursor: "pointer", opacity: filter === "all" ? 1 : 0.62 }}
        >
          <b>{jobs.length}</b>
          <span>all</span>
        </button>
        {STATUS_ORDER.map((status) => (
          <button
            key={status}
            className="count"
            data-kind={status}
            onClick={() => setFilter(status)}
            style={{ cursor: "pointer", opacity: filter === status ? 1 : 0.62 }}
          >
            <b>{counts[status] ?? 0}</b>
            <span>{status}</span>
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {visible.length === 0 ? (
        <div className="panel">
          <div className="empty">
            No {filter === "all" ? "" : filter} jobs yet. Queue work from the Tools tab, the
            Timeline tab, or an attached MCP client.
          </div>
        </div>
      ) : (
        <div className="job-list">
          {visible.map((job) => (
            <div
              key={job.job_id}
              className="job"
              data-status={job.status}
              data-selected={job.job_id === selectedId}
              onClick={() => onSelect(job)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === "Enter" && onSelect(job)}
            >
              <div className="job-head">
                <span className="job-tool">{job.tool}</span>
                <span className="badge" data-status={job.status}>
                  {job.status}
                </span>
                <span className="job-id">{job.job_id.slice(0, 8)}</span>
                {showProject && <span className="job-project">{job.project}</span>}
                <span className="spacer" />
                {(job.status === "queued" || job.status === "running") && (
                  <button
                    className="btn small danger"
                    disabled={busy === job.job_id}
                    onClick={(event) => onCancel(job, event)}
                  >
                    {busy === job.job_id ? "Cancelling…" : "Cancel"}
                  </button>
                )}
              </div>

              <div className="progress">
                <i style={{ width: `${Math.max(2, job.progress)}%` }} />
              </div>

              <div className="job-meta">
                <span>{job.message ?? "—"}</span>
                <span className="spacer" />
                <span>{job.progress.toFixed(0)}%</span>
                <span>{formatElapsed(job)}</span>
              </div>

              {job.error && (
                <div className="job-error">
                  <b>{job.error.code}</b>: {job.error.message}
                </div>
              )}

              {job.outputs.length > 0 && (
                <div className="job-meta">
                  <span className="mono">
                    {job.outputs.map((path) => basename(path)).join(", ")}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
