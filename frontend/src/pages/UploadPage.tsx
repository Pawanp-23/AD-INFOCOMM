import { FileAudio, FileVideo, ShieldCheck, UploadCloud, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";

import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Button, Card, CardHeader } from "@/components/ui/primitives";
import { api } from "@/lib/api";

const ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.mp4,.webm,.mov,.mkv,.avi,.flv";
const MAX_MB = 200;

export function UploadPage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [consent, setConsent] = useState(false);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  function pick(f: File | undefined) {
    setError(null);
    if (!f) return;
    if (f.size > MAX_MB * 1024 * 1024) return setError(`File is larger than ${MAX_MB} MB.`);
    const ext = `.${f.name.split(".").pop()?.toLowerCase()}`;
    if (!ACCEPT.split(",").includes(ext)) return setError("Unsupported format. Use an audio or video file.");
    setFile(f);
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files[0]);
  }

  async function submit() {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const session = await api.createSession("upload", title || file.name);
      await api.upload(session.id, file, setProgress);
      navigate(`/sessions/${session.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setProgress(null);
    }
  }

  const isVideo = file && /\.(mp4|webm|mov|mkv|avi|flv)$/i.test(file.name);

  return (
    <>
      <PageHeader title="Upload recording" description="Analyse a recorded call or video. Results appear segment by segment as they are processed." />
      <PageBody className="max-w-3xl">
        <Card>
          <CardHeader title="Recording" description={`Audio or video, up to ${MAX_MB} MB / 60 minutes`} />
          <div className="space-y-5 p-4">
            {!file ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
                className={`flex flex-col items-center rounded-lg border border-dashed px-6 py-10 text-center transition-colors ${drag ? "border-accent bg-accent/5" : "border-border"}`}
              >
                <UploadCloud className="h-6 w-6 text-muted" aria-hidden />
                <p className="mt-3 text-sm font-medium">Drag and drop a file here</p>
                <p className="mt-1 text-xs text-muted">WAV, MP3, FLAC, M4A, MP4, WEBM, MOV…</p>
                <Button className="mt-4" size="sm" onClick={() => input.current?.click()}>
                  Choose file
                </Button>
                <input ref={input} type="file" accept={ACCEPT} className="sr-only" aria-label="Choose a recording" onChange={(e) => pick(e.target.files?.[0])} />
              </div>
            ) : (
              <div className="flex items-center gap-3 rounded border border-border bg-subtle px-3 py-2.5">
                {isVideo ? <FileVideo className="h-5 w-5 text-accent" aria-hidden /> : <FileAudio className="h-5 w-5 text-accent" aria-hidden />}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{file.name}</div>
                  <div className="text-xs text-muted">
                    {(file.size / 1024 / 1024).toFixed(1)} MB · {isVideo ? "voice, words and face" : "voice and words"}
                  </div>
                </div>
                {progress === null && (
                  <button onClick={() => setFile(null)} className="rounded p-1 text-muted hover:bg-surface hover:text-fg" aria-label="Remove file">
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
            )}

            <div>
              <label htmlFor="title" className="mb-1.5 block text-xs font-medium">
                Title
              </label>
              <input
                id="title"
                value={title}
                maxLength={200}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Billing escalation — account 4471"
                className="h-9 w-full rounded border border-border bg-surface px-3 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
              />
            </div>

            <label className="flex cursor-pointer items-start gap-3 rounded border border-border p-3">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[rgb(var(--accent))]" />
              <span className="text-xs leading-relaxed">
                <span className="flex items-center gap-1.5 font-medium text-fg">
                  <ShieldCheck className="h-3.5 w-3.5 text-accent" aria-hidden /> Everyone in this recording consented to emotion analysis
                </span>
                <span className="text-muted">
                  The file is deleted once analysis finishes; only derived results are stored. Faces are used for expression only, never identity.
                </span>
              </span>
            </label>

            {error && (
              <p role="alert" className="rounded border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                {error}
              </p>
            )}
            {progress !== null && (
              <div aria-live="polite">
                <div className="mb-1 flex justify-between text-xs text-muted">
                  <span>{progress < 100 ? "Uploading…" : "Validating media…"}</span>
                  <span className="num">{progress}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-subtle">
                  <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <Button variant="primary" icon={UploadCloud} disabled={!file || !consent} loading={progress !== null} onClick={submit}>
                Analyse recording
              </Button>
            </div>
          </div>
        </Card>
      </PageBody>
    </>
  );
}
