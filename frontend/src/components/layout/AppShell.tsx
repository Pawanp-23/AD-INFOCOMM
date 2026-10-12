import { clsx } from "clsx";
import {
  Activity,
  Cpu,
  LayoutDashboard,
  ListVideo,
  LogOut,
  Menu,
  Moon,
  Radio,
  Sun,
  Upload,
  X,
} from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { useTheme } from "@/hooks/useTheme";
import { useAuth } from "@/lib/auth";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/sessions", label: "Sessions", icon: ListVideo },
  { to: "/live", label: "Live analysis", icon: Radio },
  { to: "/upload", label: "Upload recording", icon: Upload },
  { to: "/system", label: "Model & system", icon: Cpu, minRole: "analyst" as const },
];
const RANK = { agent: 1, analyst: 2, admin: 3 };

export function Logo() {
  return (
    <div className="flex items-center gap-2">
      <div className="grid h-7 w-7 place-items-center rounded-md bg-accent text-accent-fg">
        <Activity className="h-4 w-4" aria-hidden />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight">RAMEI</div>
        <div className="text-2xs text-faint">Emotion Intelligence</div>
      </div>
    </div>
  );
}

export function AppShell() {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const items = NAV.filter((n) => !n.minRole || RANK[user?.role ?? "agent"] >= RANK[n.minRole]);

  const nav = (
    <nav aria-label="Main" className="flex flex-col gap-0.5">
      {items.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            clsx(
              "flex items-center gap-2.5 rounded px-2.5 py-1.5 text-sm transition-colors",
              isActive ? "bg-subtle font-medium text-fg" : "text-muted hover:bg-subtle hover:text-fg",
            )
          }
        >
          <Icon className="h-4 w-4" aria-hidden />
          {label}
        </NavLink>
      ))}
    </nav>
  );

  const footer = (
    <div className="border-t border-border pt-3">
      <div className="flex items-center gap-2.5 px-1">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-subtle text-xs font-semibold uppercase text-muted">
          {(user?.full_name ?? user?.email ?? "?").slice(0, 2)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-medium">{user?.full_name ?? user?.email}</div>
          <div className="text-2xs capitalize text-faint">{user?.role}</div>
        </div>
        <button
          onClick={toggle}
          className="rounded p-1.5 text-muted hover:bg-subtle hover:text-fg"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
        <button onClick={logout} className="rounded p-1.5 text-muted hover:bg-subtle hover:text-fg" aria-label="Sign out">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[232px_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen flex-col gap-6 border-r border-border bg-surface px-3 py-4 lg:flex">
        <div className="px-1">
          <Logo />
        </div>
        <div className="flex-1">{nav}</div>
        {footer}
      </aside>
      {/* Mobile top bar */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-surface px-4 py-2.5 lg:hidden">
        <Logo />
        <button onClick={() => setOpen(true)} className="rounded p-1.5 text-muted hover:bg-subtle" aria-label="Open menu">
          <Menu className="h-5 w-5" />
        </button>
      </header>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Menu">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 flex w-64 flex-col gap-6 bg-surface px-3 py-4">
            <div className="flex items-center justify-between px-1">
              <Logo />
              <button onClick={() => setOpen(false)} className="rounded p-1.5 text-muted hover:bg-subtle" aria-label="Close menu">
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-1">{nav}</div>
            {footer}
          </div>
        </div>
      )}
      <main id="main" key={location.pathname} className="min-w-0">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  breadcrumb,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  breadcrumb?: React.ReactNode;
}) {
  return (
    <div className="border-b border-border bg-surface">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-end sm:justify-between sm:px-6">
        <div className="min-w-0">
          {breadcrumb && <div className="mb-1 text-xs text-muted">{breadcrumb}</div>}
          <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>
          {description && <p className="mt-0.5 text-sm text-muted">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
      </div>
    </div>
  );
}

export function PageBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={clsx("mx-auto max-w-7xl px-4 py-6 sm:px-6", className)}>{children}</div>;
}
