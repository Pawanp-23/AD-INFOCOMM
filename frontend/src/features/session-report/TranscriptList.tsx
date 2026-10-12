import { clsx } from "clsx";
import { Split } from "lucide-react";
import { useEffect, useRef } from "react";

import { emotionColor, emotionLabel } from "@/lib/emotions";
import { clock, pct } from "@/lib/format";
import type { Segment } from "@/types/api";

/** Transcript lines coloured by emotion (colour + text label, never colour alone). */
export function TranscriptList({
  segments,
  selectedId,
  onSelect,
  follow = false,
}: {
  segments: Segment[];
  selectedId?: string;
  onSelect: (s: Segment) => void;
  follow?: boolean;
}) {
  const end = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (follow) end.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [follow, segments.length]);

  return (
    <ul className="divide-y divide-border" role="list">
      {segments.map((s) => (
        <li key={s.id}>
          <button
            onClick={() => onSelect(s)}
            aria-current={selectedId === s.id}
            className={clsx(
              "grid w-full grid-cols-[52px_1fr] gap-3 border-l-2 px-4 py-2.5 text-left transition-colors hover:bg-subtle",
              selectedId === s.id ? "bg-subtle" : "",
            )}
            style={{ borderLeftColor: emotionColor(s.emotion) }}
          >
            <span className="num pt-0.5 font-mono text-2xs text-faint">{clock(s.start_ms)}</span>
            <span className="min-w-0">
              <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
                <span className="font-medium">{emotionLabel(s.emotion)}</span>
                <span className="num text-faint">{pct(s.confidence)}</span>
                {s.conflict && (
                  <span className="flex items-center gap-1 text-info">
                    <Split className="h-3 w-3" aria-hidden />
                    mixed cues
                  </span>
                )}
              </span>
              <span className={clsx("mt-0.5 block truncate text-xs", s.transcript ? "text-muted" : "italic text-faint")}>
                {s.transcript || "No words recognized"}
              </span>
            </span>
          </button>
        </li>
      ))}
      <li ref={end} aria-hidden />
    </ul>
  );
}
