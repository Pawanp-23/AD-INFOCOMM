"""Test configuration: an isolated SQLite database per test session, set before app import."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="ramei-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_tmp / "uploads")
os.environ["MODEL_BUNDLE_DIR"] = str(_tmp / "bundles")  # no model in unit tests
os.environ["JWT_SECRET"] = "test-secret-not-for-production-use-0123456789"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.security import hash_password  # noqa: E402
from app.db.models import Organization, User  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402

PASSWORD = "Correct-Horse-9"


@pytest.fixture(scope="session")
def client() -> TestClient:
    init_db()
    with SessionLocal() as db:
        org = Organization(name="Test Org")
        other = Organization(name="Other Org")
        db.add_all([org, other])
        db.flush()
        for email, role, org_id in [
            ("admin@test.io", "admin", org.id),
            ("agent1@test.io", "agent", org.id),
            ("agent2@test.io", "agent", org.id),
            ("outsider@test.io", "admin", other.id),
        ]:
            db.add(
                User(org_id=org_id, email=email, role=role, password_hash=hash_password(PASSWORD))
            )
        db.commit()
    with TestClient(app) as c:
        yield c


def auth(client: TestClient, email: str) -> dict[str, str]:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
