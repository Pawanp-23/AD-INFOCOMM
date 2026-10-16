import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BellRing, Check, Download, FileJson, Loader2, MessageSquareText, Sparkles, Trash2, TrendingDown, TrendingUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { EmotionLegend, EmotionTimeline, ModalityLegend, ReliabilityChart, ValenceArousalChart } from "@/components/charts/charts";
import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Badge, Button, Card, CardHeader, EmotionChip, EmptyState, ErrorState, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { SegmentInspector } from "@/features/session-report/SegmentInspector";
import { TranscriptList } from "@/features/session-report/TranscriptList";
import { api } from "@/lib/api";
import { clock, dateTime, duration } from "@/lib/format";
import type { Segment } from "@/types/api";

type ChartTab = "reliability" | "va";

export function SessionReportPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [selected, setSelected] = useState<Segment | null>(null);
  const [tab, setTab] = useState<ChartTab>("reliability");

  const session = useQuery({
    queryKey: ["session", id],
    queryFn: () => api.session(id),
    refetchInterval: (q) => (["processing", "queued", "pending"].includes(q.state.data?.status ?? "") ? 2500 : false),
  });
  const running = ["processing", "queued"].includes(session.data?.status ?? "");
  const segments = useQuery({
    queryKey: ["segments", id],
    queryFn: () => api.segments(id),
    refetchInterval: running ? 2500 : false,
  });
  const alerts = useQuery({ queryKey: ["alerts", id], queryFn: () => api.alerts(id), refetchInterval: running ? 5000 : false });

  useEffect(() => {
    if (!running) {
      qc.invalidateQueries({ queryKey: ["segments", id] });
      qc.invalidateQueries({ queryKey: ["alerts", id] });
    }
  }, [running, id, qc]);

  const ack = useMutation({
    mutationFn: (aid: string) => api.ackAlert(id, aid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts", id] }),
  });
  const del = useMutation({
    mutationFn: () => api.deleteSession(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      navigate("/sessions");
    },
  });

  const segs = useMemo(() => segments.data ?? [], [segments.data]);
  const current = selected ?? segs.find((s) => !s.abstained) ?? segs[0] ?? null;

  if (session.isError) {
    return (
      <PageBody>
        <Card>
          <ErrorState message={(session.error as Error).message} onRetry={() => session.refetch()} />
        </Card>
      </PageBody>
    );
  }
  const s = session.data;
  const summary = s?.summary;

  return (
    <>
      <PageHeader
        breadcrumb={
          <Link to="/sessions" className="hover:text-fg">
            Sessions
          </Link>
        }
        title={s?.title ?? (session.isLoading ? "Loading…" : "Untitled session")}
        description={s ? `${s.mode === "live" ? "Live session" : "Uploaded recording"} · ${dateTime(s.created_at)} · ${duration(s.duration_ms)}` : undefined}
        actions={
          s && (
            <>
              <StatusBadge status={s.status} />
              <Button size="sm" icon={Download} disabled={!segs.length} onClick={() => api.download(id, "csv")}>
                CSV
              </Button>
              <Button size="sm" icon={FileJson} disabled={!segs.length} onClick={() => api.download(id, "json")}>
                JSON
              </Button>
              <Button
                size="sm"
                variant="danger"
                icon={Trash2}
                disabled={running}
                loading={del.isPending}
                onClick={() => confirm("Delete this session and all its results? This cannot be undone.") && del.mutate()}
              >
                Delete
              </Button>
            </>
          )
        }
      />
      <PageBody className="space-y-6">
        {s && running && (
          <Card className="flex items-center gap-4 p-4" aria-live="polite">
            <Loader2 className="h-5 w-5 animate-spin text-accent" aria-hidden />
            <div className="flex-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{s.status === "queued" ? "Queued — loading models" : "Analysing recording"}</span>
                <span className="num text-muted">{s.progress}%</span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-subtle">
                <div className="h-full rounded-full bg-accent transition-[width] duration-700" style={{ width: `${s.progress}%` }} />
              </div>
              <p className="mt-1.5 text-xs text-muted">Segments appear below as soon as they are analysed. The first run also loads the models (about a minute).</p>
            </div>
          </Card>
        )}
        {s?.status === "failed" && (
          <Card className="border-danger/40 p-4">
            <p className="text-sm font-medium text-danger">Analysis failed</p>
            <p className="mt-1 text-xs text-muted">{s.error_message}</p>
          </Card>
        )}

        {summary && (
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-accent" aria-hidden /> Session summary
                </span>
              }
              action={
                summary.trend && (
                  <Badge tone={summary.trend === "worsening" ? "danger" : summary.trend === "improving" ? "success" : "neutral"}>
                    {summary.trend === "worsening" ? <TrendingDown className="h-3 w-3" /> : summary.trend === "improving" ? <TrendingUp className="h-3 w-3" /> : null}
                    Mood {summary.trend}
                  </Badge>
                )
              }
            />
            <div className="grid gap-4 p-4 md:grid-cols-[220px_1fr]">
              <div>
                <div className="label">Dominant emotion</div>
                <div className="mt-1.5">
                  <EmotionChip emotion={summary.dominant} size="lg" />
                </div>
                <p className="mt-2 text-sm text-muted">{summary.headline}</p>
              </div>
              <div>
                {summary.narrative && summary.narrative_source && summary.narrative_source !== "rules" && (
                  <p className="mb-3 text-sm">
                    {summary.narrative}
                    <span className="ml-2 text-xs text-faint">AI-written from session statistics</span>
                  </p>
                )}
                <ul className="space-y-1.5 text-sm">
                {summary.points.map((p) => (
                  <li key={p} className="flex gap-2">
                    <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-faint" aria-hidden />
                    {p}
                  </li>
                ))}
                </ul>
              </div>
            </div>
          </Card>
        )}

        <Card>
          <CardHeader title="Emotion timeline" description="Fused, calibrated probabilities per segment. Click to inspect; shaded spans are alerts." action={<EmotionLegend />} />
          <div className="p-3">
            {segments.isLoading ? (
              <Skeleton className="h-[220px]" />
            ) : segs.length ? (
              <EmotionTimeline segments={segs} alerts={alerts.data} selected={current} onSelect={setSelected} />
            ) : (
              <EmptyState icon={MessageSquareText} title={running ? "Waiting for the first segment" : "No segments"} description={running ? "Analysis is running." : "No speech or faces were detected in this recording."} />
            )}
          </div>
        </Card>

        {segs.length > 0 && (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
            <div className="space-y-6">
              <Card>
                <CardHeader
                  title={tab === "reliability" ? "Modality reliability" : "Valence and arousal"}
                  description={tab === "reliability" ? "How much each signal could be trusted over time" : "Valence (solid) and arousal (dashed), estimated from category probabilities"}
                  action={
                    <div role="tablist" className="flex gap-1 rounded border border-border p-0.5">
                      {(["reliability", "va"] as ChartTab[]).map((t) => (
                        <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`rounded px-2.5 py-1 text-xs font-medium ${tab === t ? "bg-subtle text-fg" : "text-muted"}`}>
                          {t === "reliability" ? "Reliability" : "Valence/arousal"}
                        </button>
                      ))}
                    </div>
                  }
                />
                <div className="p-3">
                  {tab === "reliability" ? (
                    <>
                      <ReliabilityChart segments={segs} />
                      <div className="px-2 pt-1">
                        <ModalityLegend />
                      </div>
                    </>
                  ) : (
                    <ValenceArousalChart segments={segs} />
                  )}
                </div>
              </Card>

              <Card>
                <CardHeader title="Transcript" description={`${segs.length} segments · coloured by detected emotion`} />
                <div className="max-h-[480px] overflow-y-auto">
                  <TranscriptList segments={segs} selectedId={current?.id} onSelect={setSelected} follow={running} />
                </div>
              </Card>
            </div>

            <div className="space-y-6">
              <SegmentInspector segment={current} />
              <Card>
                <CardHeader title="Alerts" description="Raised by the session rules" />
                {alerts.data?.length ? (
                  <ul className="divide-y divide-border">
                    {alerts.data.map((a) => (
                      <li key={a.id} className="flex items-start gap-3 px-4 py-3">
                        <BellRing className={`mt-0.5 h-4 w-4 shrink-0 ${a.severity === "critical" ? "text-danger" : a.severity === "warning" ? "text-warning" : "text-info"}`} aria-hidden />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm">{a.message}</p>
                          <button
                            className="num text-xs text-accent hover:underline"
                            onClick={() => setSelected(segs.find((x) => x.start_ms >= a.start_ms) ?? null)}
                          >
                            {clock(a.start_ms)} – {clock(a.end_ms ?? a.start_ms)}
                          </button>
                        </div>
                        {a.acknowledged ? (
                          <Badge tone="success">
                            <Check className="h-3 w-3" /> Ack
                          </Badge>
                        ) : (
                          <Button size="sm" variant="ghost" onClick={() => ack.mutate(a.id)}>
                            Acknowledge
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="px-4 py-5 text-xs text-muted">No alerts for this session.</p>
                )}
              </Card>
            </div>
          </div>
        )}
      </PageBody>
    </>
  );
}
