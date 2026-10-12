import { describe, expect, it } from "vitest";

import { EMOTION_COLOR, EMOTIONS, emotionColor, emotionLabel } from "../src/lib/emotions";
import { clock, duration, pct } from "../src/lib/format";

describe("format", () => {
  it("formats clock times", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(65_000)).toBe("1:05");
    expect(clock(3_725_000)).toBe("1:02:05");
  });

  it("formats durations", () => {
    expect(duration(null)).toBe("—");
    expect(duration(42_000)).toBe("42s");
    expect(duration(125_000)).toBe("2m 5s");
    expect(duration(3_900_000)).toBe("1h 5m");
  });

  it("formats percentages", () => {
    expect(pct(0.873)).toBe("87%");
    expect(pct(0.873, 1)).toBe("87.3%");
    expect(pct(undefined)).toBe("—");
  });
});

describe("emotion palette", () => {
  it("covers every emotion with a distinct colour and a label", () => {
    const colors = EMOTIONS.map((e) => EMOTION_COLOR[e]);
    expect(new Set(colors).size).toBe(EMOTIONS.length);
    EMOTIONS.forEach((e) => expect(emotionLabel(e)).toMatch(/^[A-Z]/));
  });

  it("falls back for unknown or missing emotions", () => {
    expect(emotionLabel(null)).toBe("Uncertain");
    expect(emotionColor("not-an-emotion")).toBe(EMOTION_COLOR.uncertain);
  });
});
