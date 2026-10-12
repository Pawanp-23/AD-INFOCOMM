import { clsx } from "clsx";
import { AlertTriangle, Loader2, type LucideIcon } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";

import { emotionColor, emotionLabel } from "@/lib/emotions";

/* ------------------------------------------------------------------ Button */
type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent/90 border-transparent",
  secondary: "bg-surface text-fg border-border hover:bg-subtle",
  ghost: "bg-transparent text-muted border-transparent hover:bg-subtle hover:text-fg",
  danger: "bg-surface text-danger border-border hover:bg-danger/10",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: LucideIcon;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", loading, icon: Icon, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded border font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-xs" : "h-9 px-4 text-sm",
        VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : Icon && <Icon className="h-4 w-4" aria-hidden />}
      {children}
    </button>
  );
});

/* ------------------------------------------------------------------ Card */
export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx("rounded-lg border border-border bg-surface", className)} {...rest} />;
}

export function CardHeader({
  title,
  description,
  action,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx("flex items-start justify-between gap-4 border-b border-border px-4 py-3", className)}>
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
        {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ Badge */
type Tone = "neutral" | "accent" | "success" | "warning" | "danger" | "info";
const TONES: Record<Tone, string> = {
  neutral: "bg-subtle text-muted border-border",
  accent: "bg-accent/10 text-accent border-accent/25",
  success: "bg-success/10 text-success border-success/25",
  warning: "bg-warning/10 text-warning border-warning/25",
  danger: "bg-danger/10 text-danger border-danger/25",
  info: "bg-info/10 text-info border-info/25",
};

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium leading-none",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const STATUS_TONE: Record<string, Tone> = {
  completed: "success",
  processing: "info",
  queued: "info",
  pending: "neutral",
  failed: "danger",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge tone={STATUS_TONE[status] ?? "neutral"}>
      {(status === "processing" || status === "queued") && (
        <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-current" aria-hidden />
      )}
      {status[0].toUpperCase() + status.slice(1)}
    </Badge>
  );
}

/* ------------------------------------------------------------------ Emotion chip */
export function EmotionChip({ emotion, size = "sm" }: { emotion: string | null | undefined; size?: "sm" | "lg" }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 font-medium text-fg",
        size === "lg" ? "text-lg" : "text-xs",
      )}
    >
      <span
        className={clsx("shrink-0 rounded-full", size === "lg" ? "h-3 w-3" : "h-2 w-2")}
        style={{ backgroundColor: emotionColor(emotion) }}
        aria-hidden
      />
      {emotionLabel(emotion)}
    </span>
  );
}

/* ------------------------------------------------------------------ Meter */
export function Meter({
  value,
  label,
  tone = "accent",
  muted,
}: {
  value: number;
  label: string;
  tone?: "accent" | "danger" | "warning";
  muted?: boolean;
}) {
  const color = muted ? "bg-faint/40" : tone === "danger" ? "bg-danger" : tone === "warning" ? "bg-warning" : "bg-accent";
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}
      className="h-1.5 w-full overflow-hidden rounded-full bg-subtle"
    >
      <div className={clsx("h-full rounded-full transition-[width] duration-500", color)} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

/* ------------------------------------------------------------------ Skeleton / states */
export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded bg-subtle", className)} aria-hidden />;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div role="status" className="flex flex-col items-center px-6 py-12 text-center">
      <div className="rounded-lg border border-border bg-subtle p-2.5">
        <Icon className="h-5 w-5 text-muted" aria-hidden />
      </div>
      <h3 className="mt-3 text-sm font-semibold">{title}</h3>
      <p className="mt-1 max-w-sm text-xs text-muted">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center px-6 py-10 text-center">
      <AlertTriangle className="h-5 w-5 text-danger" aria-hidden />
      <p className="mt-2 text-sm font-medium">Something went wrong</p>
      <p className="mt-1 max-w-sm text-xs text-muted">{message}</p>
      {onRetry && (
        <Button size="sm" className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-border bg-subtle px-1 font-mono text-2xs text-muted">{children}</kbd>;
}
