import { AudioLines, Eye, Lock, MessageSquareText, ShieldCheck } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { Logo } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth";

export function LoginPage() {
  const { user, login } = useAuth();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={(location.state as { from?: string } | null)?.from ?? "/"} replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-[1fr_minmax(420px,520px)]">
      <section className="relative hidden overflow-hidden border-r border-border bg-surface lg:flex lg:flex-col lg:justify-between lg:p-10">
        <Logo />
        <div className="max-w-lg">
          <h1 className="text-3xl font-semibold leading-tight tracking-tight">
            Understand how people feel —<br />
            <span className="text-accent">and know when to trust it.</span>
          </h1>
          <p className="mt-4 text-sm leading-relaxed text-muted">
            RAMEI reads emotion from voice, words and face together, scores how reliable each signal is at every
            moment, and shifts its trust to the signals that hold up. When nothing is reliable, it says so.
          </p>
          <ul className="mt-8 space-y-3 text-sm">
            {[
              { icon: AudioLines, t: "Voice", d: "WavLM speech encoder, robust to accents" },
              { icon: MessageSquareText, t: "Words", d: "Whisper transcript + conversational text model" },
              { icon: Eye, t: "Face", d: "Per-frame facial expression with quality checks" },
            ].map(({ icon: Icon, t, d }) => (
              <li key={t} className="flex items-start gap-3">
                <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-md border border-border bg-subtle">
                  <Icon className="h-3.5 w-3.5 text-accent" aria-hidden />
                </span>
                <span>
                  <span className="font-medium">{t}</span>
                  <span className="block text-xs text-muted">{d}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
        <p className="flex items-center gap-2 text-xs text-faint">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
          Consent-first. Raw recordings are deleted after analysis by default.
        </p>
      </section>

      <section className="flex items-center justify-center px-4 py-12 sm:px-8">
        <div className="w-full max-w-sm">
          <div className="mb-8 lg:hidden">
            <Logo />
          </div>
          <h2 className="text-xl font-semibold tracking-tight">Sign in</h2>
          <p className="mt-1 text-sm text-muted">Use the account your administrator created for you.</p>
          <form onSubmit={submit} className="mt-6 space-y-4" noValidate>
            <div>
              <label htmlFor="email" className="mb-1.5 block text-xs font-medium">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="username"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-9 w-full rounded border border-border bg-surface px-3 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
                placeholder="name@company.com"
              />
            </div>
            <div>
              <label htmlFor="password" className="mb-1.5 block text-xs font-medium">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="h-9 w-full rounded border border-border bg-surface px-3 text-sm focus:border-accent focus:outline-none"
              />
            </div>
            {error && (
              <p role="alert" className="rounded border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                {error}
              </p>
            )}
            <Button type="submit" variant="primary" className="w-full" loading={busy} icon={Lock} disabled={!email || !password}>
              Sign in
            </Button>
          </form>
        </div>
      </section>
    </div>
  );
}
