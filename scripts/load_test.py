"""API load test (PRD §9 / M4): concurrent authenticated users against the read endpoints.

Measures throughput, latency percentiles and error rate for the REST API. Model inference
latency (NFR-01) is measured offline by the ML benchmarks; this checks the web tier.

    python scripts/load_test.py --email admin@ramei.local --password-file backend/.dev-credentials
    python scripts/load_test.py --users 50 --seconds 30 --base http://127.0.0.1:8000/api/v1
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

ENDPOINTS = ("/health", "/auth/me", "/sessions?limit=20", "/dashboard/stats")


def read_password(path: Path, email: str) -> str:
    """backend/.dev-credentials holds 'email role password' lines written by app.cli create-user."""
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == email:
            return parts[2]
    raise SystemExit(f"No password for {email} in {path}")


async def user(client: httpx.AsyncClient, headers: dict, deadline: float, out: list) -> None:
    i = 0
    while time.perf_counter() < deadline:
        path = ENDPOINTS[i % len(ENDPOINTS)]
        t0 = time.perf_counter()
        try:
            r = await client.get(path, headers=headers)
            out.append((path, r.status_code, time.perf_counter() - t0))
        except httpx.HTTPError as exc:
            out.append((path, type(exc).__name__, time.perf_counter() - t0))
        i += 1


def pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[q - 1] * 1000 if len(values) > 1 else 0.0


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8000/api/v1")
    p.add_argument("--email", default="admin@ramei.local")
    p.add_argument("--password", help="defaults to the entry in --password-file")
    p.add_argument("--password-file", type=Path, default=Path("backend/.dev-credentials"))
    p.add_argument("--users", type=int, default=20)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--json", type=Path, help="also write the results to this file")
    args = p.parse_args()

    password = args.password or read_password(args.password_file, args.email)
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx.AsyncClient(base_url=args.base, timeout=30, limits=limits) as client:
        r = await client.post("/auth/login", json={"email": args.email, "password": password})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        results: list = []
        deadline = time.perf_counter() + args.seconds
        t0 = time.perf_counter()
        await asyncio.gather(*(user(client, headers, deadline, results) for _ in range(args.users)))
        elapsed = time.perf_counter() - t0

    ok = [d for _, s, d in results if s == 200]
    report = {
        "users": args.users,
        "seconds": round(elapsed, 1),
        "requests": len(results),
        "rps": round(len(results) / elapsed, 1),
        "error_rate": round(1 - len(ok) / max(len(results), 1), 4),
        "status_codes": dict(Counter(str(s) for _, s, _ in results)),
        "latency_ms": {
            "p50": round(pct(ok, 50), 1),
            "p95": round(pct(ok, 95), 1),
            "p99": round(pct(ok, 99), 1),
        },
        "per_endpoint_p95_ms": {
            e: round(pct([d for path, s, d in results if path == e and s == 200], 95), 1)
            for e in ENDPOINTS
        },
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
