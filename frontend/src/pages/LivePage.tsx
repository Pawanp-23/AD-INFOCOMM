import { AudioLines, BellRing, Camera, CameraOff, Circle, Eye, FileText, Loader2, MessageSquareText, Mic, Radio, ShieldCheck, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { EmotionLegend, EmotionTimeline } from "@/components/charts/charts";
import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Badge, Button, Card, CardHeader, EmotionChip, EmptyState, Meter } from "@/components/ui/primitives";
import { useLiveSession } from "@/features/live-session/useLiveSession";
import { SegmentInspector } from "@/features/session-report/SegmentInspector";
import { TranscriptList } from "@/features/session-report/TranscriptList";
import { emotionColor, MODALITIES, MODALITY_LABEL } from "@/lib/emotions";
import { clock, pct } from "@/lib/format";
import type { Modality, Segment } from "@/types/api";

const MOD_ICON: Record<Modality, typeof AudioLines> = { audio: AudioLines, text: MessageSquareText, visual: Eye };

const STATE_TEXT: Record<string, string> = {
  requesting: "Waiting for microphone/camera permission…",
  connecting: "Connecting to the analysis server…",
  loading_models: "Loading models (first session only, about a minute)…",
  stopping: "Finishing the last segment…",
};

export function LivePage() {
  const live = useLiveSession();
  const [title, setTitle] = useState("");
  const [camera, setCamera] = useState(true);
  const [consent, setConsent] = useState(false);
  const [selected, setSelected] = useState<Segment | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = live.stream;
  }, [live.stream]);

  const busy = ["requesting", "connecting", "loading_models", "stopping"].includes(live.state);
  const active = live.state === "live" || busy;
  const latest = live.segments[live.segments.length - 1] ?? null;

  return (
    <>
      <PageHeader
        title="Live analysis"
        description="Stream your microphone and camera. A new result arrives every 3 seconds."
        actions={
          live.state === "live" ? (
            <Button variant="danger" icon={Square} onClick={live.stop}>
              Stop session
            </Button>
          ) : live.state === "ended" && live.sessionId ? (
            <Link to={`/sessions/${live.sessionId}`}>
              <Button variant="primary" icon={FileText}>
                Open full report
              </Button>
            </Link>
          ) : undefined
        }
      />
      <PageBody className="space-y-6">
        {!active && (
          <Card className="max-w-2xl">
            <CardHeader title="New live session" description="Results are stored; the audio and video stream are not." />
            <div className="space-y-4 p-4">
              <div>
                <label htmlFor="live-title" className="mb-1.5 block text-xs font-medium">
                  Title
                </label>
                <input
                  id="live-title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Support call — order #8812"
                  className="h-9 w-full rounded border border-border bg-surface px-3 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
                />
              </div>
              <div className="flex gap-2">
                <Button variant={camera ? "primary" : "secondary"} size="sm" icon={Camera} onClick={() => setCamera(true)} aria-pressed={camera}>
                  Mic + camera
                </Button>
                <Button variant={!camera ? "primary" : "secondary"} size="sm" icon={Mic} onClick={() => setCamera(false)} aria-pressed={!camera}>
                  Mic only
                </Button>
              </div>
              <label className="flex cursor-pointer items-start gap-3 rounded border border-border p-3">
                <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[rgb(var(--accent))]" />
                <span className="text-xs">
                  <span className="flex items-center gap-1.5 font-medium">
                    <ShieldCheck className="h-3.5 w-3.5 text-accent" aria-hidden /> All participants consent to live emotion analysis
                  </span>
                  <span className="text-muted">Required before capture starts.</span>
                </span>
              </label>
              {live.error && (
                <p role="alert" className="rounded border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                  {live.error}
                </p>
              )}
              <div className="flex justify-end">
                <Button variant="primary" icon={Radio} disabled={!consent} onClick={() => live.start(title, camera)}>
                  Start live analysis
                </Button>
              </div>
            </div>
          </Card>
        )}

        {(active || live.state === "ended") && (
          <>
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
              <Card className="overflow-hidden">
                <div className="relative aspect-video bg-black">
                  {live.stream?.getVideoTracks().length ? (
                    <video ref={videoRef} autoPlay muted playsInline className="h-full w-full -scale-x-100 object-cover" aria-label="Your camera preview" />
                  ) : (
                    <div className="grid h-full place-items-center text-faint">
                      {live.state === "ended" ? <span className="text-sm">Session ended</span> : <CameraOff className="h-8 w-8" aria-label="Camera off" />}
                    </div>
                  )}
                  <div className="absolute left-3 top-3 flex items-center gap-2">
                    {live.state === "live" && (
                      <Badge tone="danger" className="bg-black/60">
                        <Circle className="h-2 w-2 animate-pulse-dot fill-current" aria-hidden /> LIVE · <span className="num">{clock(live.elapsed)}</span>
                      </Badge>
                    )}
                    {live.degraded && <Badge tone="warning" className="bg-black/60">Catching up</Badge>}
                  </div>
                  {latest && live.state === "live" && (
                    <div className="absolute bottom-3 left-3 rounded-md bg-black/65 px-3 py-2 text-white backdrop-blur-sm" aria-live="polite">
                      <div className="flex items-center gap-2 text-base font-semibold">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: emotionColor(latest.emotion) }} />
                        {latest.abstained ? "Uncertain" : latest.emotion[0].toUpperCase() + latest.emotion.slice(1)}
                        <span className="num text-xs font-normal text-white/70">{pct(latest.confidence)}</span>
                      </div>
                    </div>
                  )}
                  {busy && (
                    <div className="absolute inset-0 grid place-items-center bg-black/50 text-white">
                      <span className="flex items-center gap-2 text-sm">
                        <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {STATE_TEXT[live.state]}
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3 border-t border-border px-4 py-2.5">
                  <Mic className="h-4 w-4 text-muted" aria-hidden />
                  <div className="flex-1">
                    <Meter value={live.level} label="Microphone level" />
                  </div>
                  <span className="text-xs text-muted">{live.segments.length} segments</span>
                </div>
              </Card>

              <Card>
                <CardHeader title="Right now" description="Latest 3-second window" />
                {latest ? (
                  <div className="space-y-4 p-4">
                    <div className="flex items-end justify-between">
                      <EmotionChip emotion={latest.emotion} size="lg" />
                      <span className="num text-2xl font-semibold">{pct(latest.confidence)}</span>
                    </div>
                    <ul className="space-y-3">
                      {MODALITIES.map((m) => {
                        const info = latest.modalities[m];
                        const Icon = MOD_ICON[m];
                        return (
                          <li key={m} className="grid grid-cols-[76px_1fr_38px] items-center gap-2 text-xs">
                            <span className="flex items-center gap-1.5 text-muted">
                              <Icon className="h-3.5 w-3.5" aria-hidden />
                              {MODALITY_LABEL[m]}
                            </span>
                            {info.available ? (
                              <Meter value={info.weight} label={`${MODALITY_LABEL[m]} weight`} />
                            ) : (
                              <span className="text-faint">unavailable</span>
                            )}
                            <span className="num text-right">{info.available ? pct(info.weight) : ""}</span>
                          </li>
                        );
                      })}
                    </ul>
                    <p className="text-2xs text-faint">Bars show how much each signal contributed to the decision.</p>
                    {latest.explanation[0] && <p className="rounded bg-subtle px-3 py-2 text-xs">{latest.explanation[0]}</p>}
                  </div>
                ) : (
                  <EmptyState icon={AudioLines} title="Listening…" description="Speak normally. The first result appears after 3 seconds." />
                )}
                {live.alerts.length > 0 && (
                  <div className="border-t border-border p-4">
                    <div className="label mb-2">Alerts</div>
                    <ul className="space-y-2" aria-live="assertive">
                      {live.alerts.map((a) => (
                        <li key={a.id} className="flex items-center gap-2 text-xs">
                          <BellRing className="h-3.5 w-3.5 text-danger" aria-hidden />
                          {a.message}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>
            </div>

            {live.segments.length > 0 && (
              <>
                <Card>
                  <CardHeader title="Live timeline" action={<EmotionLegend />} />
                  <div className="p-3">
                    <EmotionTimeline segments={live.segments} alerts={live.alerts} selected={selected} onSelect={setSelected} height={180} />
                  </div>
                </Card>
                <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
                  <Card>
                    <CardHeader title="Transcript" />
                    <div className="max-h-[360px] overflow-y-auto">
                      <TranscriptList segments={live.segments} selectedId={selected?.id} onSelect={setSelected} follow={live.state === "live"} />
                    </div>
                  </Card>
                  <SegmentInspector segment={selected ?? latest} />
                </div>
              </>
            )}
          </>
        )}
      </PageBody>
    </>
  );
}
