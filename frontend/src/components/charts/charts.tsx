import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { EMOTION_COLOR, EMOTION_LABEL, EMOTIONS, MODALITIES, MODALITY_LABEL } from "@/lib/emotions";
import { clock, pct } from "@/lib/format";
import type { Alert, Segment } from "@/types/api";

/** SVG presentation attributes can't use CSS variables: resolve tokens and follow theme changes. */
function useTokens() {
  const read = () => {
    const s = getComputedStyle(document.documentElement);
    const v = (n: string) => `rgb(${s.getPropertyValue(`--${n}`).trim().replace(/ /g, ",")})`;
    return { grid: v("border"), axis: v("faint"), fg: v("fg"), surface: v("surface"), accent: v("accent"), danger: v("danger") };
  };
  const [tokens, setTokens] = useState(read);
  useEffect(() => {
    const obs = new MutationObserver(() => setTokens(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return tokens;
}

const MODALITY_COLOR = { audio: "#0072B2", text: "#E69F00", visual: "#009E73" } as const;

function ChartTooltip({ active, payload, label, kind }: any) {
  if (!active || !payload?.length) return null;
  const rows = [...payload].filter((p) => p.value !== null && p.value !== undefined).sort((a, b) => b.value - a.value);
  return (
    <div className="rounded border border-border bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="mb-1 font-medium text-muted">{clock(label)}</div>
      {rows.slice(0, kind === "prob" ? 4 : 6).map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="num font-medium">{kind === "va" ? p.value.toFixed(2) : pct(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

const axisProps = (color: string) => ({
  stroke: color,
  tick: { fill: color, fontSize: 11 },
  tickLine: false,
  axisLine: false,
});

/** Stacked emotion probabilities over time; alert spans shaded; selected segment highlighted. */
export function EmotionTimeline({
  segments,
  alerts = [],
  selected,
  onSelect,
  height = 220,
}: {
  segments: Segment[];
  alerts?: Alert[];
  selected?: Segment | null;
  onSelect?: (s: Segment) => void;
  height?: number;
}) {
  const t = useTokens();
  const data = useMemo(
    () => segments.map((s) => ({ t: s.start_ms, seg: s, ...s.probabilities })),
    [segments],
  );
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 0, left: -18 }}
        onClick={(e: any) => e?.activePayload?.[0] && onSelect?.(e.activePayload[0].payload.seg)}
        style={{ cursor: onSelect ? "pointer" : undefined }}
      >
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={clock} {...axisProps(t.axis)} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} {...axisProps(t.axis)} />
        {alerts.map((a) => (
          <ReferenceArea key={a.id} x1={a.start_ms} x2={a.end_ms ?? a.start_ms} fill={t.danger} fillOpacity={0.08} ifOverflow="extendDomain" />
        ))}
        {selected && <ReferenceLine x={selected.start_ms} stroke={t.fg} strokeDasharray="3 3" />}
        <Tooltip content={<ChartTooltip kind="prob" />} cursor={{ stroke: t.axis }} />
        {EMOTIONS.map((e) => (
          <Area
            key={e}
            type="monotone"
            dataKey={e}
            name={EMOTION_LABEL[e]}
            stackId="p"
            stroke={EMOTION_COLOR[e]}
            fill={EMOTION_COLOR[e]}
            fillOpacity={0.55}
            strokeWidth={1}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Valence and arousal trajectory in [-1, 1]. */
export function ValenceArousalChart({ segments, height = 180 }: { segments: Segment[]; height?: number }) {
  const t = useTokens();
  const data = segments.map((s) => ({ t: s.start_ms, valence: s.valence, arousal: s.arousal }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={clock} {...axisProps(t.axis)} />
        <YAxis domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} {...axisProps(t.axis)} />
        <ReferenceLine y={0} stroke={t.axis} />
        <Tooltip content={<ChartTooltip kind="va" />} cursor={{ stroke: t.axis }} />
        <Line type="monotone" dataKey="valence" name="Valence" stroke={t.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="arousal" name="Arousal" stroke="#D55E00" strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Per-modality reliability over time — shows the system shifting trust. */
export function ReliabilityChart({ segments, height = 180 }: { segments: Segment[]; height?: number }) {
  const t = useTokens();
  const data = segments.map((s) => ({
    t: s.start_ms,
    ...Object.fromEntries(MODALITIES.map((m) => [m, s.modalities[m].available ? s.modalities[m].reliability : null])),
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={clock} {...axisProps(t.axis)} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} {...axisProps(t.axis)} />
        <Tooltip content={<ChartTooltip kind="rel" />} cursor={{ stroke: t.axis }} />
        {MODALITIES.map((m, i) => (
          <Line
            key={m}
            type="stepAfter"
            dataKey={m}
            name={MODALITY_LABEL[m]}
            stroke={MODALITY_COLOR[m]}
            strokeWidth={2}
            strokeDasharray={["", "5 3", "2 2"][i]}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ModalityLegend() {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-muted">
      {MODALITIES.map((m, i) => (
        <span key={m} className="flex items-center gap-1.5">
          <svg width="18" height="6" aria-hidden>
            <line x1="0" y1="3" x2="18" y2="3" stroke={MODALITY_COLOR[m]} strokeWidth="2" strokeDasharray={["", "5 3", "2 2"][i]} />
          </svg>
          {MODALITY_LABEL[m]}
        </span>
      ))}
    </div>
  );
}

/** Horizontal share bars per emotion (accessible: label + value text for every bar). */
export function DistributionBars({ distribution }: { distribution: Record<string, number> }) {
  const rows = Object.entries(distribution)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  if (!rows.length) return <p className="text-xs text-muted">No labelled speech yet.</p>;
  return (
    <ul className="space-y-2.5">
      {rows.map(([e, v]) => (
        <li key={e} className="grid grid-cols-[84px_1fr_44px] items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: EMOTION_COLOR[e as keyof typeof EMOTION_COLOR] ?? EMOTION_COLOR.uncertain }} />
            {EMOTION_LABEL[e as keyof typeof EMOTION_LABEL] ?? e}
          </span>
          <div className="h-2 overflow-hidden rounded-full bg-subtle">
            <div
              className="h-full rounded-full"
              style={{ width: `${v * 100}%`, background: EMOTION_COLOR[e as keyof typeof EMOTION_COLOR] ?? EMOTION_COLOR.uncertain }}
            />
          </div>
          <span className="num text-right text-muted">{pct(v)}</span>
        </li>
      ))}
    </ul>
  );
}

export function EmotionLegend() {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
      {EMOTIONS.map((e) => (
        <span key={e} className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: EMOTION_COLOR[e] }} />
          {EMOTION_LABEL[e]}
        </span>
      ))}
    </div>
  );
}
