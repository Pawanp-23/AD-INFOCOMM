import { useQuery } from "@tanstack/react-query";
import { BellRing, ListVideo, Radio, Search, Upload } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";

import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Badge, Button, Card, EmotionChip, EmptyState, ErrorState, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { MODALITY_LABEL } from "@/lib/emotions";
import { dateTime, duration } from "@/lib/format";

const STATUSES = ["", "completed", "processing", "failed"];
const PAGE = 15;

export function SessionsPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const status = params.get("status") ?? "";
  const page = Number(params.get("page") ?? 0);
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    if (k !== "page") next.delete("page");
    setParams(next, { replace: true });
  };

  const sessions = useQuery({
    queryKey: ["sessions", q, status, page],
    queryFn: () => api.sessions({ q, status, limit: PAGE, offset: page * PAGE }),
    refetchInterval: (query) =>
      query.state.data?.items.some((s) => s.status === "processing" || s.status === "queued") ? 3000 : 20_000,
  });

  return (
    <>
      <PageHeader
        title="Sessions"
        description="Every analysed call and recording in your organisation."
        actions={
          <>
            <Link to="/upload">
              <Button icon={Upload}>Upload</Button>
            </Link>
            <Link to="/live">
              <Button variant="primary" icon={Radio}>
                Live
              </Button>
            </Link>
          </>
        }
      />
      <PageBody className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="relative flex-1 sm:max-w-xs">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-faint" aria-hidden />
            <input
              type="search"
              aria-label="Search sessions by title"
              placeholder="Search by title"
              defaultValue={q}
              onChange={(e) => set("q", e.target.value)}
              className="h-9 w-full rounded border border-border bg-surface pl-8 pr-3 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
            />
          </div>
          <div role="tablist" aria-label="Filter by status" className="flex gap-1 rounded border border-border bg-surface p-0.5">
            {STATUSES.map((s) => (
              <button
                key={s || "all"}
                role="tab"
                aria-selected={status === s}
                onClick={() => set("status", s)}
                className={`rounded px-3 py-1 text-xs font-medium ${status === s ? "bg-subtle text-fg" : "text-muted hover:text-fg"}`}
              >
                {s ? s[0].toUpperCase() + s.slice(1) : "All"}
              </button>
            ))}
          </div>
        </div>

        <Card className="overflow-hidden">
          {sessions.isError ? (
            <ErrorState message={(sessions.error as Error).message} onRetry={() => sessions.refetch()} />
          ) : sessions.isLoading ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-11" />
              ))}
            </div>
          ) : !sessions.data?.items.length ? (
            <EmptyState
              icon={ListVideo}
              title={q || status ? "No matching sessions" : "No sessions yet"}
              description={q || status ? "Try a different search or filter." : "Upload a recording or start a live session."}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-border bg-subtle/60 text-left">
                  <tr className="label">
                    <th scope="col" className="px-4 py-2 font-medium">Session</th>
                    <th scope="col" className="px-4 py-2 font-medium">Dominant</th>
                    <th scope="col" className="hidden px-4 py-2 font-medium md:table-cell">Modalities</th>
                    <th scope="col" className="hidden px-4 py-2 font-medium sm:table-cell">Duration</th>
                    <th scope="col" className="hidden px-4 py-2 font-medium lg:table-cell">Created</th>
                    <th scope="col" className="px-4 py-2 text-right font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {sessions.data.items.map((s) => (
                    <tr key={s.id} className="group hover:bg-subtle">
                      <td className="max-w-[280px] px-4 py-2.5">
                        <Link to={`/sessions/${s.id}`} className="block truncate font-medium group-hover:text-accent">
                          {s.title ?? "Untitled session"}
                        </Link>
                        <span className="flex items-center gap-2 text-xs text-muted">
                          {s.mode === "live" ? "Live" : "Upload"}
                          {s.alert_count > 0 && (
                            <span className="flex items-center gap-1 text-danger">
                              <BellRing className="h-3 w-3" aria-hidden />
                              {s.alert_count} alert{s.alert_count > 1 ? "s" : ""}
                            </span>
                          )}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">{s.dominant_emotion ? <EmotionChip emotion={s.dominant_emotion} /> : <span className="text-faint">—</span>}</td>
                      <td className="hidden px-4 py-2.5 md:table-cell">
                        <div className="flex gap-1">
                          {s.modalities_used.length ? s.modalities_used.map((m) => <Badge key={m}>{MODALITY_LABEL[m]}</Badge>) : <span className="text-faint">—</span>}
                        </div>
                      </td>
                      <td className="num hidden px-4 py-2.5 text-muted sm:table-cell">{duration(s.duration_ms)}</td>
                      <td className="hidden px-4 py-2.5 text-muted lg:table-cell">{dateTime(s.created_at)}</td>
                      <td className="px-4 py-2.5 text-right">
                        <StatusBadge status={s.status} />
                        {(s.status === "processing" || s.status === "queued") && <div className="num mt-0.5 text-2xs text-faint">{s.progress}%</div>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {sessions.data && sessions.data.total > PAGE && (
            <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-muted">
              <span className="num">
                {page * PAGE + 1}–{Math.min((page + 1) * PAGE, sessions.data.total)} of {sessions.data.total}
              </span>
              <div className="flex gap-2">
                <Button size="sm" disabled={page === 0} onClick={() => set("page", String(page - 1))}>
                  Previous
                </Button>
                <Button size="sm" disabled={(page + 1) * PAGE >= sessions.data.total} onClick={() => set("page", String(page + 1))}>
                  Next
                </Button>
              </div>
            </div>
          )}
        </Card>
      </PageBody>
    </>
  );
}
