"""Admin CLI.

    python -m app.cli create-user --email admin@ramei.local --role admin --name "RAMEI Admin"

A random password is generated (unless --password is given) and written to
backend/.dev-credentials (git-ignored) — never printed into logs or committed.
"""

import argparse
import secrets
import string

from sqlalchemy import select

from app.core.config import BACKEND_ROOT
from app.core.security import hash_password
from app.db.models import Organization, User
from app.db.session import SessionLocal, init_db

CREDENTIALS_FILE = BACKEND_ROOT / ".dev-credentials"


def generate_password(n: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(n))
        if any(c.isdigit() for c in pw) and any(c.isupper() for c in pw):
            return pw


def create_user(
    email: str, role: str, name: str | None, password: str | None, org_name: str
) -> None:
    init_db()
    password = password or generate_password()
    with SessionLocal() as db:
        org = db.scalar(select(Organization).where(Organization.name == org_name))
        if org is None:
            org = Organization(name=org_name)
            db.add(org)
            db.flush()
        user = db.scalar(select(User).where(User.email == email.lower()))
        if user is None:
            user = User(org_id=org.id, email=email.lower())
            db.add(user)
        user.full_name, user.role, user.is_active = name, role, True
        user.password_hash = hash_password(password)
        db.commit()
    lines = []
    if CREDENTIALS_FILE.exists():
        lines = [
            x
            for x in CREDENTIALS_FILE.read_text().splitlines()
            if not x.startswith(f"{email.lower()} ")
        ]
    lines.append(f"{email.lower()} {role} {password}")
    CREDENTIALS_FILE.write_text(
        "# email role password  (local development only)\n"
        + "\n".join(x for x in lines if not x.startswith("#"))
        + "\n"
    )
    print(f"User {email.lower()} ({role}) ready. Password saved to {CREDENTIALS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user")
    c.add_argument("--email", required=True)
    c.add_argument("--role", choices=["admin", "analyst", "agent"], default="analyst")
    c.add_argument("--name")
    c.add_argument("--password")
    c.add_argument("--org", default="AD-INFOCOMM")
    args = parser.parse_args()
    create_user(args.email, args.role, args.name, args.password, args.org)


if __name__ == "__main__":
    main()
