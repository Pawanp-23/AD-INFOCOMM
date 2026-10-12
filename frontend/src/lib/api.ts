import type {
  Alert,
  AnalysisSession,
  DashboardStats,
  ModelStatus,
  Page,
  Segment,
  TokenResponse,
  User,
} from "@/types/api";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api/v1";
const TOKEN_KEY = "ramei-token";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export const tokenStore = {
  get: (): string | null => {
    try {
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set: (t: string | null) => {
    try {
      if (t) sessionStorage.setItem(TOKEN_KEY, t);
      else sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* storage unavailable: token lives only in memory for this tab */
    }
  },
};

let onUnauthorized: () => void = () => {};
export const setUnauthorizedHandler = (fn: () => void) => {
  onUnauthorized = fn;
};

function messageFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => (x as { msg?: string }).msg ?? "").join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = tokenStore.get();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (res.status === 401 && !path.startsWith("/auth/login")) onUnauthorized();
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, messageFrom(body, res.statusText || "Request failed"));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  me: () => request<User>("/auth/me"),
  stats: () => request<DashboardStats>("/dashboard/stats"),
  models: () => request<ModelStatus>("/models"),
  health: () => request<{ status: string; model_bundle_found: boolean; model_loaded: boolean }>("/health"),
  sessions: (params: { status?: string; q?: string; limit?: number; offset?: number } = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && qs.set(k, String(v)));
    return request<Page<AnalysisSession>>(`/sessions?${qs}`);
  },
  session: (id: string) => request<AnalysisSession>(`/sessions/${id}`),
  createSession: (mode: "live" | "upload", title: string | undefined) =>
    request<AnalysisSession>("/sessions", {
      method: "POST",
      body: JSON.stringify({ mode, title, consent: true }),
    }),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),
  segments: (id: string) => request<Segment[]>(`/sessions/${id}/segments`),
  alerts: (id: string) => request<Alert[]>(`/sessions/${id}/alerts`),
  ackAlert: (sid: string, aid: string) =>
    request<Alert>(`/sessions/${sid}/alerts/${aid}/ack`, { method: "POST" }),
  upload: (id: string, file: File, onProgress?: (pct: number) => void) =>
    new Promise<AnalysisSession>((resolve, reject) => {
      // XHR for upload progress events (fetch has none)
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/sessions/${id}/media`);
      const token = tokenStore.get();
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(Math.round((e.loaded / e.total) * 100));
      xhr.onload = () => {
        const body = (() => {
          try {
            return JSON.parse(xhr.responseText);
          } catch {
            return null;
          }
        })();
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as AnalysisSession);
        else reject(new ApiError(xhr.status, messageFrom(body, "Upload failed")));
      };
      xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    }),
  exportUrl: (id: string, format: "json" | "csv") => `${BASE}/sessions/${id}/export?format=${format}`,
  download: async (id: string, format: "json" | "csv") => {
    const token = tokenStore.get();
    const res = await fetch(`${BASE}/sessions/${id}/export?format=${format}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, "Export failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ramei-session-${id.slice(0, 8)}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  },
};

export function streamUrl(sessionId: string): string {
  const explicit = import.meta.env.VITE_WS_URL as string | undefined;
  const base = explicit ?? `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/v1/stream`;
  return `${base}/${sessionId}?token=${encodeURIComponent(tokenStore.get() ?? "")}`;
}
