/**
 * Client for the local UI's API.
 *
 * Every call goes to the same backend the MCP server uses, so anything done
 * here is visible to an attached MCP client and vice versa.
 */

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  job_id: string;
  tool: string;
  status: JobStatus;
  progress: number;
  message: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  outputs: string[];
  result: Record<string, unknown>;
  error: { code: string; message: string; details?: Record<string, unknown> } | null;
  command: string | null;
  params: Record<string, unknown>;
  project: string;
}

export interface JobsResponse {
  jobs: Job[];
  counts: Record<string, number>;
}

/** One project and how much work it holds. */
export interface ProjectSummary {
  name: string;
  jobs: number;
  queued: number;
  running: number;
  done: number;
  failed: number;
  last_activity: number | null;
  is_active: boolean;
}

export interface ProjectList {
  active: string;
  projects: ProjectSummary[];
}

export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  const?: unknown;
  format?: string;
}

export interface Tool {
  name: string;
  title: string;
  description: string;
  phase: number;
  read_only: boolean;
  input_schema: JsonSchema;
  output_schema: JsonSchema;
}

export interface MediaFile {
  path: string;
  name: string;
  size_bytes: number;
  modified_at: number;
  kind: "video" | "audio" | "image" | "subtitle" | "other";
}

export interface Workspace {
  workspace: string;
  allowed_roots: string[];
  jobs_dir: string;
  uploads_dir: string;
}

/** The token is taken from the URL when the server was bound off-loopback. */
const token = new URLSearchParams(window.location.search).get("token");

function headers(): Record<string, string> {
  const base: Record<string, string> = { "Content-Type": "application/json" };
  if (token) base["x-auth-token"] = token;
  return base;
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string;
    try {
      const body = await response.json();
      detail =
        body?.error?.message ??
        (typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body));
    } catch {
      detail = `${response.status} ${response.statusText}`;
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<{ workspace: string; tool_count: number }> {
  return unwrap(await fetch("/api/health"));
}

export async function getWorkspace(): Promise<Workspace> {
  return unwrap(await fetch("/api/workspace", { headers: headers() }));
}

export async function listJobs(): Promise<JobsResponse> {
  return unwrap(await fetch("/api/jobs?limit=200", { headers: headers() }));
}

export async function getJob(jobId: string): Promise<Job> {
  return unwrap(await fetch(`/api/jobs/${jobId}`, { headers: headers() }));
}

export async function cancelJob(jobId: string): Promise<Job> {
  return unwrap(
    await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST", headers: headers() }),
  );
}

/**
 * Projects go through the tool endpoints rather than `/api/projects`, because
 * the tool also reports which project is active — including one switched into
 * but not yet used, which has no jobs to be listed from.
 */
export async function listProjects(): Promise<ProjectList> {
  return callTool<ProjectList>("list_projects", {});
}

/**
 * Switch the project new work is filed under.
 *
 * This is the UI process's own active project; an attached MCP client keeps
 * its own, so switching here cannot move that session's work.
 */
export async function setProject(name: string): Promise<{ output_directory: string }> {
  return callTool<{ output_directory: string }>("set_project", { name });
}

export async function listTools(): Promise<Tool[]> {
  return unwrap(await fetch("/api/tools", { headers: headers() }));
}

export async function callTool<T = Record<string, unknown>>(
  name: string,
  args: unknown,
): Promise<T> {
  return unwrap(
    await fetch(`/api/tools/${name}`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(args),
    }),
  );
}

export async function listFiles(directory?: string): Promise<MediaFile[]> {
  const query = directory ? `?directory=${encodeURIComponent(directory)}` : "";
  return unwrap(await fetch(`/api/files${query}`, { headers: headers() }));
}

export interface UploadAccepts {
  suffixes: string[];
  max_bytes: number;
}

export async function getUploadAccepts(): Promise<UploadAccepts> {
  return unwrap(await fetch("/api/upload/accepts", { headers: headers() }));
}

/**
 * Send files to the workspace.
 *
 * Lets the UI work on media from anywhere on the machine without the user
 * first copying it under an allowed root. The Content-Type header is
 * deliberately omitted so the browser sets the multipart boundary itself.
 */
export async function uploadFiles(files: File[]): Promise<MediaFile[]> {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  const init: RequestInit = { method: "POST", body };
  if (token) init.headers = { "x-auth-token": token };
  return unwrap(await fetch("/api/upload", init));
}

/** URL the <video> element loads; the server honours Range so seeking works. */
export function mediaUrl(path: string): string {
  const query = new URLSearchParams({ path });
  if (token) query.set("token", token);
  return `/api/media?${query.toString()}`;
}

export interface JobsMessage {
  type: "jobs";
  jobs: Job[];
  removed: string[];
  counts: Record<string, number>;
}

/**
 * Subscribe to job updates.
 *
 * The server pushes only changed jobs, so the queue view tracks the store
 * within a few hundred milliseconds however a job was started. Reconnects
 * automatically, since the server restarting should not leave a dead view.
 */
export function subscribeToJobs(
  onMessage: (message: JobsMessage) => void,
  onStatus?: (connected: boolean) => void,
): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: number | undefined;

  const connect = () => {
    if (closed) return;
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const query = token ? `?token=${encodeURIComponent(token)}` : "";
    socket = new WebSocket(`${scheme}://${window.location.host}/ws/jobs${query}`);
    socket.onopen = () => onStatus?.(true);
    socket.onmessage = (event) => onMessage(JSON.parse(event.data) as JobsMessage);
    socket.onclose = () => {
      onStatus?.(false);
      if (!closed) retry = window.setTimeout(connect, 1500);
    };
    socket.onerror = () => socket?.close();
  };

  connect();
  return () => {
    closed = true;
    if (retry) window.clearTimeout(retry);
    socket?.close();
  };
}
