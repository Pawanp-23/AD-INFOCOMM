import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, Cpu, XCircle } from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/AppShell";
import { Badge, Card, CardHeader, ErrorState, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { dateTime } from "@/lib/format";

const SCENARIO_LABEL: Record<string, string> = {
  clean: "Clean audio, words and face",
  audio_noise_10db: "Noisy room (10 dB SNR)",
  noisy_room_0db: "Very noisy room (0 dB SNR)",
  visual_blur: "Blurred face",
  visual_dark: "Poor lighting",
  visual_occluded: "Half of face covered",
  text_word_drop: "Half the words missed by ASR",
  camera_off: "Camera off",
  audio_missing: "Microphone off",
  noisy_room_camera_off: "Very noisy room + camera off",
  audio_only: "Voice only",
  visual_only: "Face only",
};

type Stat = { mean: number; std: number };

export function SystemPage() {
  const q = useQuery({ queryKey: ["models"], queryFn: api.models, refetchInterval: 10_000 });
  const m = q.data?.manifest;
  const cv = m?.cv_summary as Record<string, Record<string, Stat>> | null | undefined;

  return (
    <>
      <PageHeader title="Model & system" description="The model version serving predictions, its calibration and its measured robustness." />
      <PageBody className="space-y-6">
        {q.isError ? (
          <Card>
            <ErrorState message={(q.error as Error).message} onRetry={() => q.refetch()} />
          </Card>
        ) : q.isLoading ? (
          <Skeleton className="h-40" />
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ["Active model", q.data!.active_model],
                ["Bundle", q.data!.bundle_found ? "Found" : "Missing"],
                ["Loaded", q.data!.loaded ? `Yes · ${q.data!.device}` : "On first use"],
                ["Running jobs", String(q.data!.running_jobs)],
              ].map(([k, v]) => (
                <Card key={k} className="p-4">
                  <div className="label">{k}</div>
                  <div className="mt-1.5 flex items-center gap-1.5 truncate text-sm font-medium">
                    {k === "Bundle" && (q.data!.bundle_found ? <CheckCircle2 className="h-4 w-4 text-success" /> : <XCircle className="h-4 w-4 text-danger" />)}
                    {v}
                  </div>
                </Card>
              ))}
            </div>

            {m && (
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
                <Card>
                  <CardHeader
                    title="Robustness under real-world conditions"
                    description="Speaker-independent cross-validation on CREMA-D (91 actors). Macro-F1 over 7 emotions; ECE after calibration."
                  />
                  {cv ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="border-b border-border bg-subtle/60 text-left">
                          <tr className="label">
                            <th className="px-4 py-2 font-medium">Condition</th>
                            <th className="px-4 py-2 text-right font-medium">Macro-F1</th>
                            <th className="px-4 py-2 text-right font-medium">ECE</th>
                            <th className="hidden px-4 py-2 font-medium md:table-cell">Trust (voice / words / face)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {Object.entries(cv).map(([k, v]) => {
                            const w = v.weights as unknown as Record<string, number>;
                            return (
                              <tr key={k}>
                                <td className="px-4 py-2">{SCENARIO_LABEL[k] ?? k}</td>
                                <td className="num px-4 py-2 text-right font-medium">
                                  {v.macro_f1.mean.toFixed(3)} <span className="text-2xs text-faint">± {v.macro_f1.std.toFixed(3)}</span>
                                </td>
                                <td className="num px-4 py-2 text-right text-muted">{v.ece.mean.toFixed(3)}</td>
                                <td className="hidden px-4 py-2 md:table-cell">
                                  <div className="flex h-2 w-40 overflow-hidden rounded-full bg-subtle" role="img" aria-label={`weights voice ${w.audio}, words ${w.text}, face ${w.visual}`}>
                                    <div style={{ width: `${w.audio * 100}%`, background: "#0072B2" }} />
                                    <div style={{ width: `${w.text * 100}%`, background: "#E69F00" }} />
                                    <div style={{ width: `${w.visual * 100}%`, background: "#009E73" }} />
                                  </div>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="p-4 text-xs text-muted">No cross-validation results attached to this bundle.</p>
                  )}
                </Card>
                <Card>
                  <CardHeader title={<span className="flex items-center gap-2"><Cpu className="h-4 w-4 text-accent" /> {m.name}</span>} description={`Built ${dateTime(m.created_at)} · commit ${m.git_commit}`} />
                  <dl className="divide-y divide-border text-xs">
                    {Object.entries(m.encoders).map(([k, v]) => (
                      <div key={k} className="grid grid-cols-[72px_1fr] gap-3 px-4 py-2">
                        <dt className="capitalize text-muted">{k}</dt>
                        <dd className="break-all font-mono text-2xs">{v}</dd>
                      </div>
                    ))}
                    <div className="grid grid-cols-[72px_1fr] gap-3 px-4 py-2">
                      <dt className="text-muted">Temperature</dt>
                      <dd className="num">{m.temperature.toFixed(3)}</dd>
                    </div>
                    <div className="grid grid-cols-[72px_1fr] gap-3 px-4 py-2">
                      <dt className="text-muted">Abstain</dt>
                      <dd>
                        confidence &lt; {m.thresholds.abstain_confidence} or max reliability &lt; {m.thresholds.abstain_reliability}
                      </dd>
                    </div>
                    <div className="grid grid-cols-[72px_1fr] gap-3 px-4 py-2">
                      <dt className="text-muted">Training</dt>
                      <dd>
                        {m.train.dataset} · {m.train.speakers} speakers ({m.train.val_speakers} held out) · best epoch {m.train.best_epoch}
                      </dd>
                    </div>
                  </dl>
                  <div className="border-t border-border p-4">
                    <Badge tone="warning">Assistive use only</Badge>
                    <p className="mt-2 text-2xs text-muted">Trained on acted English speech. Not for clinical use or automated decisions about people.</p>
                  </div>
                </Card>
              </div>
            )}
          </>
        )}
      </PageBody>
    </>
  );
}
