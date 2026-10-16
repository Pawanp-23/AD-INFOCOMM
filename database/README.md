# database/

RAMEI's persistent data model.

```
database/
└── schema/schema.sql   # human-readable reference DDL for PostgreSQL 16
```

## How the schema is managed

- **Source of truth:** the SQLAlchemy models in [`backend/app/db/models`](../backend/app/db/models/__init__.py).
  The API creates any missing tables on startup (`init_db`), for SQLite (default) and PostgreSQL alike.
- **`schema/schema.sql`** documents the same tables, indexes and constraints for reviewers and DBAs.
  Update it in the same change as the models.
- **Users** are created with the backend CLI, never with SQL seeds:
  `python -m app.cli create-user --email admin@ramei.local --role admin`.
- Alembic migrations are the production path once the schema must evolve on live data
  (`pip install -e "backend[postgres]"` ships Alembic); see [ADR-0008](../docs/adr/0008-local-first-implementation-choices.md).

## Rules

- Never commit real user data, transcripts or recordings (raw media is not persisted by default).
- Local database files (`backend/ramei.db*`) are git-ignored.

Entity relationships and retention policy: [`docs/03-System-Architecture.md` §8](../docs/03-System-Architecture.md#8-data-architecture).
