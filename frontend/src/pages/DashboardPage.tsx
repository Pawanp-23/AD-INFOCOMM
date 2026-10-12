import { useQuery } from "@tanstack/react-query";
import { AlertOctagon, AudioLines, BellRing, Clock3, Eye, ListVideo, MessageSquareText, Radio, Upload } from "lucide-react";
import { Link } from "react-router-dom";

import { DistributionBars } from "@/components/charts/charts";
import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Badge, Button, Card, CardHeader, EmotionChip, EmptyState, ErrorState, Meter, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { MODALITY_LABEL } from "@/lib/emotions";
import { ago, clock, duration, pct } from "@/lib/format";
import type { Modality } from "@/types/api";

const MOD_ICON: Record<Modality, typeof AudioLines> = { audio: AudioLines, text: MessageSquareText, visual: Eye };

function Stat({ label, value, hint, icon: Icon }: { label: string; value: string; hint?: string; icon: typeof Clock3 }) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        <Icon className="h-4 w-4 text-faint" aria-hidden />
      </div>
      <div className="num mt-2 text-2xl font-semibold tracking-tight">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </Card>
  );
}

export function DashboardPage() {
  const { user } = useAuth();
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 15_000 });
  const recent = useQuery({ queryKey: ["sessions", "recent"], queryFn: () => api.sessions({ limit: 6 }), refetchInterval: 10_000 });

  const greeting = new Date().getHours() < 12 ? "Good morning" : new Date().getHours() < 18 ? "Good afternoon" : "Good evening";

  return (
    <>
      <PageHeader
        title={`${greeting}${user?.full_name ? `, ${user.full_name.split(" ")[0]}` : ""}`}
        description="Emotion trends, alerts and signal reliability across your sessions."
        actions={
          <>
            <Link to="/upload">
              <Button icon={Upload}>Upload recording</Button>
            </Link>
            <Link to="/live">
              <Button variant="primary" icon={Radio}>
                Start live analysis
              </Button>
            </Link>
          </>
        }
      />
      <PageBody className="space-y-6">
        {stats.isError ? (
          <Card>
            <ErrorState message={(stats.error as Error).message} onRetry={() => stats.refetch()} />
          </Card>
        ) : (
          <section aria-label="Key metrics" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {stats.isLoading || !stats.data ? (
              Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[98px]" />)
            ) : (
              <>
                <Stat label="Sessions" value={String(stats.data.sessions_total)} hint={`${stats.data.sessions_completed} completed`} icon={ListVideo} />
                <Stat label="Speech analysed" value={`${(stats.data.hours_analyzed * 60).toFixed(1)} min`} hint="across all sessions" icon={Clock3} />
                <Stat label="Alerts" value={String(stats.data.alerts_total)} hint={`${stats.data.critical_alerts} critical`} icon={BellRing} />
                <Stat
                  label="Negative share"
                  value={pct(["angry", "sad", "fear", "disgust"].reduce((a, e) => a + (stats.data!.emotion_distribution[e] ?? 0), 0))}
                  hint="of labelled speech time"
                  icon={AlertOctagon}
                />
              </>
            )}
          </section>
        )}

        <div className="grid gap-6 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader
              title="Recent sessions"
              action={
                <Link to="/sessions" className="text-xs font-medium text-accent hover:underline">
                  View all
                </Link>
              }
            />
            {recent.isLoading ? (
              <div className="space-y-2 p-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-10" />
                ))}
              </div>
            ) : recent.data?.items.length ? (
              <ul className="divide-y divide-border">
                {recent.data.items.map((s) => (
                  <li key={s.id}>
                    <Link to={`/sessions/${s.id}`} className="grid grid-cols-[1fr_auto] items-center gap-4 px-4 py-3 hover:bg-subtle sm:grid-cols-[1fr_120px_90px_100px]">
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium">{s.title ?? "Untitled session"}</span>
                        <span className="text-xs text-muted">
                          {s.mode === "live" ? "Live" : "Upload"} · {ago(s.created_at)}
                        </span>
                      </span>
                      <span className="hidden sm:block">{s.dominant_emotion ? <EmotionChip emotion={s.dominant_emotion} /> : <span className="text-xs text-faint">—</span>}</span>
                      <span className="num hidden text-xs text-muted sm:block">{duration(s.duration_ms)}</span>
                      <span className="justify-self-end">
                        <StatusBadge status={s.status} />
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                icon={ListVideo}
                title="No sessions yet"
                description="Upload a call recording or start a live session to see emotion analytics here."
                action={
                  <Link to="/upload">
                    <Button variant="primary" icon={Upload}>
                      Upload a recording
                    </Button>
                  </Link>
                }
              />
            )}
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader title="Emotion mix" description="Share of labelled speech time" />
              <div className="p-4">
                {stats.data ? <DistributionBars distribution={stats.data.emotion_distribution} /> : <Skeleton className="h-32" />}
              </div>
            </Card>
            <Card>
              <CardHeader title="Signal reliability" description="Average trust per modality" />
              <ul className="space-y-3 p-4">
                {(["audio", "text", "visual"] as Modality[]).map((m) => {
                  const Icon = MOD_ICON[m];
                  const v = stats.data?.mean_reliability[m] ?? 0;
                  return (
                    <li key={m} className="grid grid-cols-[88px_1fr_40px] items-center gap-3 text-xs">
                      <span className="flex items-center gap-1.5 text-muted">
                        <Icon className="h-3.5 w-3.5" aria-hidden />
                        {MODALITY_LABEL[m]}
                      </span>
                      <Meter value={v} label={`${MODALITY_LABEL[m]} average reliability`} />
                      <span className="num text-right">{pct(v)}</span>
                    </li>
                  );
                })}
              </ul>
            </Card>
          </div>
        </div>

        <Card>
          <CardHeader title="Recent alerts" description="Escalations detected by the alert rules" />
          {stats.data?.recent_alerts.length ? (
            <ul className="divide-y divide-border">
              {stats.data.recent_alerts.map((a) => (
                <li key={a.id}>
                  <Link to={`/sessions/${a.session_id}`} className="flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-subtle">
                    <Badge tone={a.severity === "critical" ? "danger" : a.severity === "warning" ? "warning" : "info"}>{a.severity}</Badge>
                    <span className="min-w-0 flex-1 truncate">{a.message}</span>
                    <span className="hidden truncate text-xs text-muted sm:block">{a.session_title}</span>
                    <span className="num text-xs text-faint">at {clock(a.start_ms)}</span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="px-4 py-6 text-xs text-muted">No alerts. Sustained anger, prolonged negative mood and unreliable stretches are flagged here.</p>
          )}
        </Card>
      </PageBody>
    </>
  );
}
