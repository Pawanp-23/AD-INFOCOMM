import { useCallback, useEffect, useRef, useState } from "react";

import { api, streamUrl } from "@/lib/api";
import type { Alert, Segment } from "@/types/api";

export type LiveState = "idle" | "requesting" | "connecting" | "loading_models" | "live" | "stopping" | "ended" | "error";

// AudioWorklet: forwards 128-sample frames from the mic to the main thread.
// Same-origin file in public/ (a blob: URL would need a weaker CSP).
const WORKLET_URL = "/audio-tap-worklet.js";
const SAMPLE_RATE = 16_000;
const CHUNK = 4_000; // 250 ms of audio per message
const FPS = 6;

function frame(kind: number, payload: ArrayBuffer): ArrayBuffer {
  const out = new Uint8Array(payload.byteLength + 1);
  out[0] = kind;
  out.set(new Uint8Array(payload), 1);
  return out.buffer;
}

/** Live capture + streaming protocol (TRD §7.3). */
export function useLiveSession() {
  const [state, setState] = useState<LiveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [level, setLevel] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [degraded, setDegraded] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const ws = useRef<WebSocket | null>(null);
  const ctx = useRef<AudioContext | null>(null);
  const timer = useRef<number | null>(null);
  const clockTimer = useRef<number | null>(null);
  const media = useRef<MediaStream | null>(null);

  const cleanup = useCallback(() => {
    if (timer.current) window.clearInterval(timer.current);
    if (clockTimer.current) window.clearInterval(clockTimer.current);
    timer.current = clockTimer.current = null;
    media.current?.getTracks().forEach((t) => t.stop());
    media.current = null;
    setStream(null);
    ctx.current?.close().catch(() => {});
    ctx.current = null;
  }, []);

  useEffect(() => () => {
    ws.current?.close();
    cleanup();
  }, [cleanup]);

  const start = useCallback(
    async (title: string, camera: boolean) => {
      setError(null);
      setSegments([]);
      setAlerts([]);
      setDegraded(false);
      setElapsed(0);
      try {
        setState("requesting");
        const m = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false },
          video: camera ? { width: 640, height: 480, facingMode: "user" } : false,
        });
        media.current = m;
        setStream(m);

        setState("connecting");
        const session = await api.createSession("live", title || "Live session");
        setSessionId(session.id);
        const socket = new WebSocket(streamUrl(session.id));
        socket.binaryType = "arraybuffer";
        ws.current = socket;

        socket.onmessage = (ev) => {
          const msg = JSON.parse(ev.data as string);
          if (msg.type === "status") {
            if (msg.state === "loading_models") setState("loading_models");
            if (msg.state === "ready") setState("live");
            if (msg.state === "degraded") setDegraded(true);
            if (msg.state === "closed") setState("ended");
          } else if (msg.type === "segment_result") {
            setSegments((prev) => [...prev, msg as Segment]);
          } else if (msg.type === "alert") {
            setAlerts((prev) => [...prev, { id: `${msg.rule}-${msg.start_ms}`, acknowledged: false, ...msg } as Alert]);
          } else if (msg.type === "error") {
            setError(String(msg.detail));
          }
        };
        socket.onerror = () => setError("Connection to the analysis server failed.");
        socket.onclose = () => {
          cleanup();
          setState((s) => (s === "error" ? s : "ended"));
        };
        await new Promise<void>((resolve, reject) => {
          socket.onopen = () => resolve();
          setTimeout(() => reject(new Error("Timed out connecting to the server")), 10_000);
        });
        socket.send(JSON.stringify({ type: "start" }));

        // Audio: AudioContext at 16 kHz resamples the microphone for us.
        const audio = new AudioContext({ sampleRate: SAMPLE_RATE });
        ctx.current = audio;
        await audio.audioWorklet.addModule(WORKLET_URL);
        const src = audio.createMediaStreamSource(m);
        const tap = new AudioWorkletNode(audio, "ramei-tap");
        let buf = new Float32Array(0);
        tap.port.onmessage = (e: MessageEvent<Float32Array>) => {
          const next = new Float32Array(buf.length + e.data.length);
          next.set(buf);
          next.set(e.data, buf.length);
          buf = next;
          if (buf.length >= CHUNK) {
            const chunk = buf.subarray(0, CHUNK);
            buf = buf.slice(CHUNK);
            let sum = 0;
            const pcm = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
              const v = Math.max(-1, Math.min(1, chunk[i]));
              pcm[i] = v * 0x7fff;
              sum += v * v;
            }
            setLevel(Math.min(1, Math.sqrt(sum / chunk.length) * 6));
            if (socket.readyState === WebSocket.OPEN) socket.send(frame(0x01, pcm.buffer));
          }
        };
        src.connect(tap);

        // Video: JPEG frames from a hidden canvas.
        if (camera) {
          const video = document.createElement("video");
          video.srcObject = m;
          video.muted = true;
          await video.play();
          const canvas = document.createElement("canvas");
          canvas.width = 640;
          canvas.height = 480;
          const g = canvas.getContext("2d")!;
          timer.current = window.setInterval(() => {
            if (socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 2_000_000) return;
            g.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => blob?.arrayBuffer().then((b) => socket.readyState === WebSocket.OPEN && socket.send(frame(0x02, b))),
              "image/jpeg",
              0.75,
            );
          }, 1000 / FPS);
        }
        const t0 = Date.now();
        clockTimer.current = window.setInterval(() => setElapsed(Date.now() - t0), 500);
      } catch (e) {
        cleanup();
        ws.current?.close();
        setState("error");
        setError(
          e instanceof DOMException && e.name === "NotAllowedError"
            ? "Microphone/camera permission was denied. Allow access in the browser and try again."
            : e instanceof Error
              ? e.message
              : "Could not start the session",
        );
      }
    },
    [cleanup],
  );

  const stop = useCallback(() => {
    setState("stopping");
    if (timer.current) window.clearInterval(timer.current);
    if (clockTimer.current) window.clearInterval(clockTimer.current);
    media.current?.getTracks().forEach((t) => t.stop());
    ctx.current?.close().catch(() => {});
    ctx.current = null;
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ type: "stop" }));
    else setState("ended");
  }, []);

  return { state, error, sessionId, segments, alerts, level, stream, degraded, elapsed, start, stop };
}
