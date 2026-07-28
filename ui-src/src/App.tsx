/**
 * The UI shell.
 *
 * Job state lives here and is shared by every tab, so submitting work from the
 * Tools or Timeline tab is reflected in the Jobs tab immediately, and jobs
 * started by an attached MCP client appear in exactly the same place.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getHealth,
  listJobs,
  listTools,
  subscribeToJobs,
  type Job,
  type Tool,
} from "./api";
import JobsPanel from "./components/JobsPanel";
import PreviewPanel from "./components/PreviewPanel";
import ToolsPanel from "./components/ToolsPanel";
import TimelineEditor from "./components/TimelineEditor";

type Tab = "jobs" | "preview" | "tools" | "timeline";

const TABS: [Tab, string][] = [
  ["jobs", "Jobs"],
  ["preview", "Preview"],
  ["tools", "Tools"],
  ["timeline", "Timeline"],
];

export default function App() {
  const [tab, setTab] = useState<Tab>("jobs");
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [tools, setTools] = useState<Tool[]>([]);
  const [workspace, setWorkspace] = useState<string>("");
  const [live, setLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await listJobs();
      setJobs(Object.fromEntries(response.jobs.map((job) => [job.job_id, job])));
      setCounts(response.counts);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const [health, toolList] = await Promise.all([getHealth(), listTools()]);
        setWorkspace(health.workspace);
        setTools(toolList);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
    void refresh();
  }, [refresh]);

  // Live updates: the server pushes only what changed.
  useEffect(
    () =>
      subscribeToJobs(
        (message) => {
          setJobs((current) => {
            const next = { ...current };
            for (const job of message.jobs) next[job.job_id] = job;
            for (const id of message.removed) delete next[id];
            return next;
          });
          setCounts(message.counts);
        },
        (connected) => setLive(connected),
      ),
    [],
  );

  const jobList = useMemo(
    () => Object.values(jobs).sort((a, b) => b.created_at - a.created_at),
    [jobs],
  );
  const selectedJob = selectedId ? (jobs[selectedId] ?? null) : null;

  const onQueued = useCallback(() => {
    void refresh();
    setTab("jobs");
  }, [refresh]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          ffmpeg-mcp
          <small>local editor</small>
        </div>

        <nav className="tabs">
          {TABS.map(([key, label]) => (
            <button
              key={key}
              className="tab"
              data-active={tab === key}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="topbar-right">
          {workspace && <span className="mono">{workspace}</span>}
          <span className="link-status" data-live={live} title={live ? "Live" : "Reconnecting…"}>
            <span className="pip" />
            {live ? "live" : "offline"}
          </span>
        </div>
      </header>

      <main>
        {error && <div className="error-banner">{error}</div>}

        {tab === "jobs" && (
          <JobsPanel
            jobs={jobList}
            counts={counts}
            selectedId={selectedId}
            onSelect={(job) => {
              setSelectedId(job.job_id);
              if (job.status === "done") setTab("preview");
            }}
          />
        )}
        {tab === "preview" && <PreviewPanel jobs={jobList} selectedJob={selectedJob} />}
        {tab === "tools" && <ToolsPanel tools={tools} onQueued={onQueued} />}
        {tab === "timeline" && <TimelineEditor onQueued={onQueued} />}
      </main>
    </div>
  );
}
