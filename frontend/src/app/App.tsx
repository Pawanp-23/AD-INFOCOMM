import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api";
import { AuthProvider, useAuth } from "@/lib/auth";
import { LoginPage } from "@/pages/LoginPage";

// Route-level code splitting (TRD §9)
const DashboardPage = lazy(() => import("@/pages/DashboardPage").then((m) => ({ default: m.DashboardPage })));
const SessionsPage = lazy(() => import("@/pages/SessionsPage").then((m) => ({ default: m.SessionsPage })));
const SessionReportPage = lazy(() => import("@/pages/SessionReportPage").then((m) => ({ default: m.SessionReportPage })));
const UploadPage = lazy(() => import("@/pages/UploadPage").then((m) => ({ default: m.UploadPage })));
const LivePage = lazy(() => import("@/pages/LivePage").then((m) => ({ default: m.LivePage })));
const SystemPage = lazy(() => import("@/pages/SystemPage").then((m) => ({ default: m.SystemPage })));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (n, err) => !(err instanceof ApiError && [401, 403, 404].includes(err.status)) && n < 2,
      refetchOnWindowFocus: false,
    },
  },
});

function Loading() {
  return (
    <div className="grid min-h-[50vh] place-items-center" aria-busy="true">
      <Loader2 className="h-5 w-5 animate-spin text-muted" aria-label="Loading" />
    </div>
  );
}

function RequireAuth({ children, minRole }: { children: ReactNode; minRole?: "analyst" | "admin" }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <Loading />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  const rank = { agent: 1, analyst: 2, admin: 3 };
  if (minRole && rank[user.role] < rank[minRole]) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function NotFound() {
  return (
    <div className="grid min-h-[60vh] place-items-center text-center">
      <div>
        <p className="num text-4xl font-semibold text-faint">404</p>
        <p className="mt-2 text-sm text-muted">This page does not exist.</p>
        <Link to="/">
          <Button className="mt-4" size="sm">
            Back to overview
          </Button>
        </Link>
      </div>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                element={
                  <RequireAuth>
                    <AppShell />
                  </RequireAuth>
                }
              >
                <Route index element={<DashboardPage />} />
                <Route path="sessions" element={<SessionsPage />} />
                <Route path="sessions/:id" element={<SessionReportPage />} />
                <Route path="upload" element={<UploadPage />} />
                <Route path="live" element={<LivePage />} />
                <Route
                  path="system"
                  element={
                    <RequireAuth minRole="analyst">
                      <SystemPage />
                    </RequireAuth>
                  }
                />
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
