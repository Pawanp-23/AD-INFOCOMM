# frontend/

RAMEI web dashboard — **React 18 + Vite 5 + TypeScript**, Tailwind CSS + shadcn/ui, React Router.
Spec: [TRD §9](../docs/02-TRD.md#9-frontend-specification) · Screens: [PRD §9](../docs/01-PRD.md#9-user-experience-overview).

```
frontend/
├── public/                     # static assets copied as-is
├── index.html                  # Vite entry HTML
├── src/
│   ├── main.tsx                # React root
│   ├── app/                    # app shell: App.tsx, router.tsx, providers (Query, theme, auth), route guards
│   ├── pages/                  # route screens: Login, Dashboard, LiveSession, Upload, SessionReport, Admin
│   ├── components/
│   │   ├── ui/                 # shadcn/ui primitives
│   │   ├── charts/             # EmotionTimeline, ValenceArousalChart, ReliabilityBars
│   │   └── layout/             # shell, sidebar, header
│   ├── features/               # feature modules (components + hooks + logic per feature)
│   │   ├── live-session/
│   │   ├── session-report/
│   │   ├── upload/
│   │   ├── explain/
│   │   └── admin/
│   ├── hooks/                  # useStream, useMediaCapture, useAuth
│   ├── lib/                    # api-client, ws-client, env, utils
│   ├── services/               # typed API calls per resource
│   ├── store/                  # Zustand stores (live session state)
│   ├── styles/                 # globals, theme tokens
│   └── types/                  # generated from backend OpenAPI — do not edit by hand
└── tests/                      # Vitest + React Testing Library; Playwright e2e
```

## Rules

- Pages stay thin: they compose feature modules. Feature code lives in `features/<name>`;
  only truly shared pieces go in `components/`.
- API types are generated from the backend OpenAPI schema (`openapi-typescript`), never hand-written.
- Only `VITE_*` environment variables reach the browser — never put secrets in them.
- Emotion colors must always be paired with a text label (color-blind safety, WCAG 2.1 AA).

## Configuration

| Variable | Example |
|---|---|
| `VITE_API_URL` | `http://localhost:8000/api/v1` |
| `VITE_WS_URL` | `ws://localhost:8000/api/v1/stream` |

In development the Vite dev server proxies `/api` (HTTP and WebSocket) to the backend.
In production `vite build` produces `dist/`, served by the Nginx proxy with SPA fallback.

## Initialize (milestone M3)

```bash
npm create vite@5 . -- --template react-ts
npm install react-router-dom @tanstack/react-query zustand recharts
```
