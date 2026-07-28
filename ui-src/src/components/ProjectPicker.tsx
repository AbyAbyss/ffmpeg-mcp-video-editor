/**
 * Which project the UI is working in.
 *
 * Picking a name does two things, because in a single-user local UI they are
 * the same intent: it scopes what the Jobs and Preview tabs show, and it sets
 * the project new work is filed under. "All projects" is a view only — it
 * leaves the active project alone, since there is no such place to write to.
 */

import { useState } from "react";
import type { ProjectSummary } from "../api";

interface Props {
  projects: ProjectSummary[];
  active: string;
  view: string;
  onSelect: (name: string) => void;
  onCreate: (name: string) => void;
}

export default function ProjectPicker({ projects, active, view, onSelect, onCreate }: Props) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");

  const submit = () => {
    const name = draft.trim();
    if (name) onCreate(name);
    setDraft("");
    setAdding(false);
  };

  return (
    <div className="project-picker">
      <span className="project-label">project</span>

      {adding ? (
        <input
          className="project-new"
          autoFocus
          value={draft}
          placeholder="new project name"
          onChange={(event) => setDraft(event.target.value)}
          onBlur={submit}
          onKeyDown={(event) => {
            if (event.key === "Enter") submit();
            if (event.key === "Escape") {
              setDraft("");
              setAdding(false);
            }
          }}
        />
      ) : (
        <>
          <select
            className="project-select"
            value={view}
            onChange={(event) => onSelect(event.target.value)}
          >
            <option value="all">All projects</option>
            {projects.map((project) => (
              <option key={project.name} value={project.name}>
                {project.name}
                {project.name === active ? " ●" : ""} ({project.jobs})
              </option>
            ))}
          </select>
          <button
            className="project-add"
            title="Start a new project"
            onClick={() => setAdding(true)}
          >
            +
          </button>
        </>
      )}
    </div>
  );
}
