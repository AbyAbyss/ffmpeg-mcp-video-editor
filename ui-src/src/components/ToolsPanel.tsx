/**
 * Phase 6b: run any tool directly, without an AI client in the loop.
 *
 * The forms are generated from each tool's JSON schema, so this list stays
 * complete as tools are added and never drifts from what they accept.
 */

import { useMemo, useState } from "react";
import { callTool, type Tool } from "../api";
import SchemaForm, { type Values } from "./SchemaForm";

const PHASE_NAMES: Record<number, string> = {
  1: "Core",
  2: "Colour & text",
  3: "Transcription",
  4: "Vision",
  5: "Composition",
};

interface Props {
  tools: Tool[];
  onQueued: () => void;
}

export default function ToolsPanel({ tools, onQueued }: Props) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Tool | null>(null);
  const [values, setValues] = useState<Values>({});
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const matching = needle
      ? tools.filter(
          (tool) =>
            tool.name.includes(needle) ||
            tool.title.toLowerCase().includes(needle) ||
            tool.description.toLowerCase().includes(needle),
        )
      : tools;
    const grouped = new Map<number, Tool[]>();
    for (const tool of matching) {
      const list = grouped.get(tool.phase) ?? [];
      list.push(tool);
      grouped.set(tool.phase, list);
    }
    return [...grouped.entries()].sort((a, b) => a[0] - b[0]);
  }, [tools, search]);

  const choose = (tool: Tool) => {
    setSelected(tool);
    setValues({});
    setResult(null);
    setError(null);
  };

  const submit = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await callTool(selected.name, values);
      setResult(response);
      onQueued();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid-side">
      <div className="panel">
        <h2 className="panel-title">Tools</h2>
        <input
          className="tool-search"
          type="text"
          placeholder="Search…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{ marginBottom: 12 }}
        />
        <div className="tool-list">
          {filtered.map(([phase, group]) => (
            <div key={phase}>
              <div
                className="panel-title"
                style={{ margin: "10px 0 4px", fontSize: 11 }}
              >
                {PHASE_NAMES[phase] ?? `Phase ${phase}`}
              </div>
              {group.map((tool) => (
                <div
                  key={tool.name}
                  className="tool-item"
                  data-active={selected?.name === tool.name}
                  onClick={() => choose(tool)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => event.key === "Enter" && choose(tool)}
                >
                  {tool.title}
                  <small>{tool.name}</small>
                </div>
              ))}
            </div>
          ))}
          {filtered.length === 0 && <div className="empty">No tools match.</div>}
        </div>
      </div>

      <div className="panel">
        {!selected ? (
          <div className="empty">Pick a tool to run it directly.</div>
        ) : (
          <>
            <h2 className="panel-title">
              {selected.title}
              <span className="job-id">{selected.name}</span>
            </h2>
            <p className="desc">{selected.description}</p>

            {error && <div className="error-banner">{error}</div>}

            <SchemaForm schema={selected.input_schema} values={values} onChange={setValues} />

            <div className="row" style={{ marginTop: 6 }}>
              <button className="btn primary" onClick={() => void submit()} disabled={busy}>
                {busy ? "Running…" : selected.read_only ? "Run" : "Queue job"}
              </button>
              <button className="btn" onClick={() => setValues({})} disabled={busy}>
                Reset
              </button>
            </div>

            {result !== null && (
              <details className="raw" open>
                <summary>Result</summary>
                <pre className="command">{JSON.stringify(result, null, 2)}</pre>
              </details>
            )}

            <details className="raw">
              <summary>Arguments being sent</summary>
              <pre className="command">{JSON.stringify(values, null, 2)}</pre>
            </details>
          </>
        )}
      </div>
    </div>
  );
}
