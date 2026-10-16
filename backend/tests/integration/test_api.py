"""API integration tests: auth, RBAC, org isolation, sessions lifecycle (TRD §7.2, §11)."""

from fastapi.testclient import TestClient

from tests.conftest import PASSWORD, auth


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["model_bundle_found"] is False


def test_login_success_and_me(client: TestClient) -> None:
    h = auth(client, "admin@test.io")
    me = client.get("/api/v1/auth/me", headers=h).json()
    assert me["email"] == "admin@test.io" and me["role"] == "admin"


def test_login_rejects_wrong_password(client: TestClient) -> None:
    r = client.post("/api/v1/auth/login", json={"email": "admin@test.io", "password": "nope"})
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")


def test_endpoints_require_auth(client: TestClient) -> None:
    assert client.get("/api/v1/sessions").status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.get("/api/v1/auth/me", headers=bad).status_code == 401


def test_session_requires_consent(client: TestClient) -> None:
    h = auth(client, "agent1@test.io")
    r = client.post("/api/v1/sessions", json={"mode": "upload", "consent": False}, headers=h)
    assert r.status_code == 422


def test_session_lifecycle_and_isolation(client: TestClient) -> None:
    a1, a2 = auth(client, "agent1@test.io"), auth(client, "agent2@test.io")
    admin, outsider = auth(client, "admin@test.io"), auth(client, "outsider@test.io")
    r = client.post(
        "/api/v1/sessions", headers=a1, json={"mode": "upload", "title": "Call 1", "consent": True}
    )
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["status"] == "pending"

    assert client.get(f"/api/v1/sessions/{sid}", headers=a1).status_code == 200
    assert client.get(f"/api/v1/sessions/{sid}", headers=a2).status_code == 404  # other agent
    assert client.get(f"/api/v1/sessions/{sid}", headers=admin).status_code == 200  # same org
    assert client.get(f"/api/v1/sessions/{sid}", headers=outsider).status_code == 404  # other org

    listed = client.get("/api/v1/sessions", headers=a1).json()
    assert any(s["id"] == sid for s in listed["items"])
    others = client.get("/api/v1/sessions", headers=a2).json()["items"]
    assert all(s["id"] != sid for s in others)

    bad = client.post(
        f"/api/v1/sessions/{sid}/media",
        headers=a1,
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad.status_code == 415
    fake = client.post(
        f"/api/v1/sessions/{sid}/media",
        headers=a1,
        files={"file": ("call.wav", b"RIFF-not-really-audio", "audio/wav")},
    )
    assert fake.status_code == 422  # content is validated, not just the extension

    assert client.get(f"/api/v1/sessions/{sid}/segments", headers=a1).json() == []
    csv = client.get(f"/api/v1/sessions/{sid}/export?format=csv", headers=a1)
    assert csv.status_code == 200 and csv.text.startswith("start_ms,end_ms,emotion")

    assert client.delete(f"/api/v1/sessions/{sid}", headers=a2).status_code == 404
    assert client.delete(f"/api/v1/sessions/{sid}", headers=a1).status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}", headers=a1).status_code == 404


def test_dashboard_stats(client: TestClient) -> None:
    r = client.get("/api/v1/dashboard/stats", headers=auth(client, "admin@test.io"))
    assert r.status_code == 200
    assert {"sessions_total", "emotion_distribution", "mean_reliability"} <= set(r.json())


def test_models_endpoint_requires_analyst(client: TestClient) -> None:
    assert client.get("/api/v1/models", headers=auth(client, "agent1@test.io")).status_code == 403
    r = client.get("/api/v1/models", headers=auth(client, "admin@test.io"))
    assert r.status_code == 200 and r.json()["bundle_found"] is False


def test_password_is_hashed_with_argon2(client: TestClient) -> None:
    from sqlalchemy import select

    from app.db.models import User
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        u = db.scalar(select(User).where(User.email == "admin@test.io"))
    assert u.password_hash.startswith("$argon2id$") and PASSWORD not in u.password_hash


def test_security_headers(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Cache-Control"] == "no-store"
