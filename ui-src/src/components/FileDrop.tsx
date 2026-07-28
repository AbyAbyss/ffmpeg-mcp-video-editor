/**
 * Drop target for bringing media into the workspace.
 *
 * The tools can only read paths inside the configured allowed roots, so
 * without this the UI can only edit files the user has already copied there by
 * hand. Dropping a file uploads it and hands back its workspace path.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { getUploadAccepts, uploadFiles, type MediaFile } from "../api";

interface Props {
  onUploaded: (files: MediaFile[]) => void;
  compact?: boolean;
}

export default function FileDrop({ onUploaded, compact }: Props) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accepts, setAccepts] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const depth = useRef(0);

  useEffect(() => {
    void getUploadAccepts()
      .then((info) => setAccepts(info.suffixes))
      .catch(() => setAccepts([]));
  }, []);

  const send = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setBusy(true);
      setError(null);
      try {
        onUploaded(await uploadFiles(Array.from(files)));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [onUploaded],
  );

  return (
    <div
      className="dropzone"
      data-over={over}
      data-compact={compact ? "true" : undefined}
      // dragenter/leave fire for child elements too, so nesting is counted
      // rather than toggled, or the highlight flickers as the pointer moves.
      onDragEnter={(event) => {
        event.preventDefault();
        depth.current += 1;
        setOver(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        depth.current -= 1;
        if (depth.current <= 0) setOver(false);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        depth.current = 0;
        setOver(false);
        void send(event.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={accepts.join(",")}
        hidden
        onChange={(event) => void send(event.target.files)}
      />
      {busy ? "Uploading…" : over ? "Drop to add" : "Drop media here, or click to browse"}
      {error && <div className="drop-error">{error}</div>}
    </div>
  );
}
