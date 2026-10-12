import { clsx } from "clsx";
import { AudioLines, Eye, Info, MessageSquareText, ShieldAlert, Split } from "lucide-react";

import { Badge, Card, CardHeader, EmotionChip, Meter } from "@/components/ui/primitives";
import { EMOTION_COLOR, EMOTION_LABEL, EMOTIONS, MODALITIES, MODALITY_LABEL } from "@/lib/emotions";
import { clock, pct } from "@/lib/format";
import type { Modality, Segment } from "@/types/api";

const ICON: Record<Modality, typeof AudioLines> = { audio: AudioLines, text: MessageSquareText, visual: Eye };

/** Explain panel (FR-17): what was decided, how confident, which modality was trusted and why. */
export function SegmentInspector({ segment }: { segment: Segment | null }) {
  if (!segment) {
    return (
      <Card>
        <CardHeader title="Segment details" description="Select a point on the timeline or a transcript line" />
        <div className="flex items-center gap-2 px-4 py-8 text-xs text-muted">
          <Info className="h-4 w-4" aria-hidden /> Nothing selected
        </div>
      </Card>
    );
  }
  const s = segment;
  return (
    <Card aria-live="polite">
      <CardHeader
        title={`Segment ${s.index + 1}`}
        description={`${clock(s.start_ms)} – ${clock(s.end_ms)}`}
        action={
          <div className="flex gap-1.5">
            {s.abstained && <Badge tone="warning">Abstained</Badge>}
            {s.conflict && (
              <Badge tone="info">
                <Split className="h-3 w-3" aria-hidden /> Conflict
              </Badge>
            )}
          </div>
        }
      />
      <div className="space-y-5 p-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="label">Fused prediction</div>
            <div className="mt-1">
              <EmotionChip emotion={s.emotion} size="lg" />
            </div>
          </div>
          <div className="text-right">
            <div className="label">Calibrated confidence</div>
            <div className="num mt-1 text-lg font-semibold">{pct(s.confidence)}</div>
          </div>
        </div>

        <div>
          <div className="label mb-2">Probabilities</div>
          <div className="flex h-2.5 overflow-hidden rounded-full bg-subtle" role="img" aria-label="Emotion probability breakdown">
            {EMOTIONS.map((e) => (
              <div key={e} style={{ width: `${s.probabilities[e] * 100}%`, background: EMOTION_COLOR[e] }} title={`${EMOTION_LABEL[e]} ${pct(s.probabilities[e])}`} />
            ))}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
            {[...EMOTIONS]
              .sort((a, b) => s.probabilities[b] - s.probabilities[a])
              .slice(0, 4)
              .map((e) => (
                <span key={e} className="flex justify-between gap-2">
                  <span className="text-muted">{EMOTION_LABEL[e]}</span>
                  <span className="num font-medium">{pct(s.probabilities[e])}</span>
                </span>
              ))}
          </div>
        </div>

        <div>
          <div className="label mb-2">Modalities · reliability and fusion weight</div>
          <ul className="space-y-3">
            {MODALITIES.map((m) => {
              const info = s.modalities[m];
              const Icon = ICON[m];
              return (
                <li key={m} className={clsx("rounded border border-border p-2.5", !info.available && "opacity-60")}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-xs font-medium">
                      <Icon className="h-3.5 w-3.5 text-muted" aria-hidden />
                      {MODALITY_LABEL[m]}
                    </span>
                    {info.available ? (
                      <span className="flex items-center gap-2 text-xs">
                        <span className="text-muted">says</span>
                        <EmotionChip emotion={info.emotion} />
                      </span>
                    ) : (
                      <Badge>Unavailable</Badge>
                    )}
                  </div>
                  {info.available && (
                    <div className="mt-2 grid grid-cols-[72px_1fr_36px] items-center gap-x-2 gap-y-1.5 text-2xs text-muted">
                      <span>Reliability</span>
                      <Meter value={info.reliability} label={`${MODALITY_LABEL[m]} reliability`} tone={info.reliability < 0.4 ? "danger" : info.reliability < 0.6 ? "warning" : "accent"} />
                      <span className="num text-right">{pct(info.reliability)}</span>
                      <span>Weight</span>
                      <Meter value={info.weight} label={`${MODALITY_LABEL[m]} fusion weight`} muted={info.weight < 0.05} />
                      <span className="num text-right">{pct(info.weight)}</span>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>

        {s.explanation.length > 0 && (
          <div>
            <div className="label mb-2">Why</div>
            <ul className="space-y-1.5 text-xs">
              {s.explanation.map((x) => (
                <li key={x} className="flex gap-2">
                  <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />
                  <span>{x}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {s.transcript && (
          <div>
            <div className="label mb-1">Transcript</div>
            <p className="rounded bg-subtle px-3 py-2 text-xs italic">“{s.transcript}”</p>
          </div>
        )}
      </div>
    </Card>
  );
}
