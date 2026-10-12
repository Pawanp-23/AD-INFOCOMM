# frontend/

RAMEI web dashboard — **React 18 + Vite 5 + TypeScript**, Tailwind CSS, React Router, TanStack
Query, Recharts. Spec: [TRD §9](../docs/02-TRD.md#9-frontend-specification) · Screens:
[PRD §9](../docs/01-PRD.md#9-user-experience-overview).

```
frontend/
├── index.html                  # entry HTML (fonts, theme bootstrap)
├── src/
│   ├── main.tsx                # React root
│   ├── app/App.tsx             # providers, router, protected routes, code splitting
│   ├── pages/                  # Login, Dashboard, Sessions, Upload, Live, SessionReport, System
│   ├── components/
│   │   ├── ui/primitives.tsx   # Button, Card, Badge, EmotionChip, Meter, Skeleton, Empty/Error states
│   │   ├── charts/charts.tsx   # emotion timeline, reliability, valence/arousal, distribution
│   │   └── layout/AppShell.tsx # sidebar shell, page header/body
│   ├── features/
│   │   ├── live-session/       # useLiveSession: mic (AudioWorklet, PCM16 @16 kHz) + camera -> WebSocket
│   │   └── session-report/     # SegmentInspector (explain panel), TranscriptList
│   ├── hooks/                  # useTheme
│   ├── lib/                    # api client, auth context, emotion palette, formatting
│   ├── styles/index.css        # design tokens (light + dark)
│   └── types/api.ts            # API types (mirror backend schemas)
└── public/                     # favicon
```

## Design system

- Neutral slate surfaces, a single teal accent, 1px borders instead of shadows, Inter + JetBrains Mono.
- Emotion colours use the Okabe–Ito colour-blind-safe palette and are always paired with a text label.
- Tokens are CSS variables (`src/styles/index.css`) with light and dark themes; Tailwind maps to them.
- Every data view has loading (skeleton), empty and error states; focus rings and reduced-motion support.

## Develop

```bash
npm install
npm run dev        # http://localhost:5173 — proxies /api (HTTP + WebSocket) to http://127.0.0.1:8000
npm run build      # type-check + production build into dist/
```

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | `/api/v1` | REST base URL |
| `VITE_WS_URL` | same host `/api/v1/stream` | WebSocket base URL |

Only `VITE_*` variables reach the browser — never put secrets in them.
