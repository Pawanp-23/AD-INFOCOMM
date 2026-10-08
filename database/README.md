# database/

Everything about RAMEI's persistent data model (PostgreSQL 16).

```
database/
├── schema/schema.sql   # human-readable reference DDL (kept in sync with migrations)
├── migrations/         # Alembic environment + versioned migrations (source of truth)
│   └── versions/
├── seeds/              # development seed data — never real users or recordings
└── docs/               # data dictionary, ER notes
```

## Rules

- **Schema changes go through Alembic migrations only.** Update `schema/schema.sql` in the same PR.
- Migrations must be reversible (`downgrade()` implemented).
- Never commit real user data, transcripts or recordings.

## Local usage

The Postgres container in the root `docker-compose.yml` loads `schema/schema.sql` and
`seeds/` automatically on first start (empty volume):

```bash
docker compose up -d postgres
```

To reset: `docker compose down -v` (removes the volume), then start again.

Entity relationships and retention policy: [`docs/03-System-Architecture.md` §8](../docs/03-System-Architecture.md#8-data-architecture).
