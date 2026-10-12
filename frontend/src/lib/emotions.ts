import type { Emotion, Modality } from "@/types/api";

export const EMOTIONS: Emotion[] = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"];

// Okabe–Ito colour-blind-safe palette. Always paired with a text label (WCAG 1.4.1).
export const EMOTION_COLOR: Record<Emotion | "uncertain", string> = {
  neutral: "#94a3b8",
  happy: "#E69F00",
  sad: "#0072B2",
  angry: "#D55E00",
  fear: "#CC79A7",
  disgust: "#009E73",
  surprise: "#56B4E9",
  uncertain: "#64748b",
};

export const EMOTION_LABEL: Record<Emotion | "uncertain", string> = {
  neutral: "Neutral",
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fear: "Fear",
  disgust: "Disgust",
  surprise: "Surprise",
  uncertain: "Uncertain",
};

export const MODALITIES: Modality[] = ["audio", "text", "visual"];
export const MODALITY_LABEL: Record<Modality, string> = { audio: "Voice", text: "Words", visual: "Face" };

export function emotionColor(e: string | null | undefined): string {
  return EMOTION_COLOR[(e ?? "uncertain") as Emotion] ?? EMOTION_COLOR.uncertain;
}
export function emotionLabel(e: string | null | undefined): string {
  return EMOTION_LABEL[(e ?? "uncertain") as Emotion] ?? "—";
}
