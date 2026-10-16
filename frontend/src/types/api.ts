// Mirrors backend/app/schemas (TRD §7.2). Keep in sync with the OpenAPI schema at /api/v1/docs.

export type Role = "admin" | "analyst" | "agent";
export type Emotion = "neutral" | "happy" | "sad" | "angry" | "fear" | "disgust" | "surprise";
export type Modality = "audio" | "text" | "visual";
export type SessionStatus = "pending" | "queued" | "processing" | "completed" | "failed";

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: Role;
  org_id: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface ModalityInfo {
  available: boolean;
  reliability: number;
  weight: number;
  emotion: Emotion | null;
  confidence: number | null;
}

export interface Segment {
  id: string;
  index: number;
  start_ms: number;
  end_ms: number;
  transcript: string | null;
  emotion: Emotion | "uncertain";
  confidence: number;
  abstained: boolean;
  valence: number | null;
  arousal: number | null;
  conflict: boolean;
  probabilities: Record<Emotion, number>;
  modalities: Record<Modality, ModalityInfo>;
  explanation: string[];
}

export interface SessionSummary {
  headline: string;
  points: string[];
  distribution: Record<string, number>;
  dominant?: string;
  trend?: "improving" | "worsening" | "stable";
  mean_valence?: number;
  mean_reliability?: Record<Modality, number>;
  /** LLM-written note when a provider is configured; otherwise equals the headline. */
  narrative?: string;
  /** "rules" or the model id that wrote the narrative. */
  narrative_source?: string;
}

export interface AnalysisSession {
  id: string;
  mode: "live" | "upload";
  title: string | null;
  status: SessionStatus;
  progress: number;
  duration_ms: number | null;
  modalities_used: Modality[];
  model_version: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  segment_count: number;
  alert_count: number;
  dominant_emotion: string | null;
  owner_email: string | null;
  summary?: SessionSummary | null;
}

export interface Page<T> {
  items: T[];
  total: number;
}

export interface Alert {
  id: string;
  rule: string;
  severity: "info" | "warning" | "critical";
  start_ms: number;
  end_ms: number | null;
  message: string;
  acknowledged: boolean;
}

export interface DashboardStats {
  sessions_total: number;
  sessions_completed: number;
  hours_analyzed: number;
  alerts_total: number;
  critical_alerts: number;
  emotion_distribution: Record<string, number>;
  mean_reliability: Record<Modality, number>;
  recent_alerts: Array<{
    id: string;
    session_id: string;
    session_title: string | null;
    rule: string;
    severity: Alert["severity"];
    message: string;
    start_ms: number;
    created_at: string;
  }>;
}

export interface ModelStatus {
  active_model: string;
  bundle_found: boolean;
  loaded: boolean;
  device: string | null;
  running_jobs: number;
  manifest: null | {
    name: string;
    version: string;
    created_at: string;
    git_commit: string;
    encoders: Record<string, string>;
    thresholds: Record<string, number>;
    temperature: number;
    train: { dataset: string; speakers: number; val_speakers: number; best_epoch: number };
    cv_summary: Record<string, Record<string, { mean: number; std: number } | Record<string, number>>> | null;
  };
}
