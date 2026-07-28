/** Small formatting helpers shared across the panels. */

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "--:--";
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[index]}`;
}

export function formatElapsed(job: {
  started_at: number | null;
  finished_at: number | null;
}): string {
  if (!job.started_at) return "--";
  const end = job.finished_at ?? Date.now() / 1000;
  const seconds = end - job.started_at;
  return seconds < 60 ? `${seconds.toFixed(1)}s` : formatDuration(seconds);
}

export function basename(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}
