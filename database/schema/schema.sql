-- =============================================================================
-- RAMEI — reference database schema (PostgreSQL 16)
-- Source of truth for structure is Alembic (database/migrations). This file is the
-- human-readable reference and must be kept in sync with migrations.
-- Spec: docs/02-TRD.md §8
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS citext;     -- case-insensitive email

-- ---------------------------------------------------------------- enums
CREATE TYPE user_role       AS ENUM ('admin', 'analyst', 'agent', 'integrator');
CREATE TYPE session_mode    AS ENUM ('live', 'upload', 'api');
CREATE TYPE session_status  AS ENUM ('pending', 'uploading', 'queued', 'processing', 'completed', 'failed');
CREATE TYPE modality        AS ENUM ('audio', 'text', 'visual');
CREATE TYPE emotion_label   AS ENUM ('neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'uncertain');
CREATE TYPE alert_severity  AS ENUM ('info', 'warning', 'critical');
CREATE TYPE media_kind      AS ENUM ('audio', 'video');

-- ---------------------------------------------------------------- tenancy & users
CREATE TABLE organizations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text        NOT NULL,
    settings    jsonb       NOT NULL DEFAULT '{"retention_days": 90, "store_raw_media": false}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id         uuid        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email          citext      NOT NULL UNIQUE,
    full_name      text,
    password_hash  text        NOT NULL,
    role           user_role   NOT NULL DEFAULT 'agent',
    is_active      boolean     NOT NULL DEFAULT true,
    last_login_at  timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- models
CREATE TABLE model_versions (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name          text        NOT NULL,
    version       text        NOT NULL,
    artifact_uri  text        NOT NULL,
    metrics       jsonb       NOT NULL DEFAULT '{}',
    is_active     boolean     NOT NULL DEFAULT false,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);
-- At most one active model per name.
CREATE UNIQUE INDEX uq_model_versions_active ON model_versions (name) WHERE is_active;

-- ---------------------------------------------------------------- sessions
CREATE TABLE sessions (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            uuid           NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id           uuid           REFERENCES users(id) ON DELETE SET NULL,
    model_version_id  uuid           REFERENCES model_versions(id),
    mode              session_mode   NOT NULL,
    title             text,
    status            session_status NOT NULL DEFAULT 'pending',
    progress          smallint       NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    duration_ms       integer        CHECK (duration_ms >= 0),
    modalities_used   modality[]     NOT NULL DEFAULT '{}',
    error_message     text,
    created_at        timestamptz    NOT NULL DEFAULT now(),
    completed_at      timestamptz
);
CREATE INDEX ix_sessions_org_created ON sessions (org_id, created_at DESC);
CREATE INDEX ix_sessions_user        ON sessions (user_id);

CREATE TABLE consents (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id            uuid        REFERENCES sessions(id) ON DELETE CASCADE,
    consent_text_version  text        NOT NULL,
    ip_hash               text,
    granted_at            timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_consents_session ON consents (session_id);

CREATE TABLE media_assets (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   uuid        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind         media_kind  NOT NULL,
    storage_uri  text        NOT NULL,
    size_bytes   bigint      NOT NULL CHECK (size_bytes >= 0),
    sha256       char(64)    NOT NULL,
    expires_at   timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_media_assets_expires ON media_assets (expires_at) WHERE expires_at IS NOT NULL;

-- ---------------------------------------------------------------- results
CREATE TABLE segments (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     uuid          NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx            integer       NOT NULL CHECK (idx >= 0),
    start_ms       integer       NOT NULL CHECK (start_ms >= 0),
    end_ms         integer       NOT NULL,
    transcript     text,
    emotion        emotion_label NOT NULL,
    confidence     real          NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    abstained      boolean       NOT NULL DEFAULT false,
    valence        real          CHECK (valence BETWEEN -1 AND 1),
    arousal        real          CHECK (arousal BETWEEN -1 AND 1),
    conflict       boolean       NOT NULL DEFAULT false,
    probabilities  jsonb         NOT NULL,
    explanation    jsonb         NOT NULL DEFAULT '[]',
    created_at     timestamptz   NOT NULL DEFAULT now(),
    CHECK (end_ms > start_ms),
    UNIQUE (session_id, idx)
);

CREATE TABLE modality_predictions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id     uuid          NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    modality       modality      NOT NULL,
    available      boolean       NOT NULL,
    reliability    real          NOT NULL CHECK (reliability BETWEEN 0 AND 1),
    weight         real          NOT NULL CHECK (weight BETWEEN 0 AND 1),
    emotion        emotion_label,
    confidence     real          CHECK (confidence BETWEEN 0 AND 1),
    probabilities  jsonb,
    quality        jsonb         NOT NULL DEFAULT '{}',
    UNIQUE (segment_id, modality)
);

CREATE TABLE alerts (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    uuid           NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    rule          text           NOT NULL,
    severity      alert_severity NOT NULL DEFAULT 'warning',
    start_ms      integer        NOT NULL,
    end_ms        integer,
    message       text           NOT NULL,
    acknowledged  boolean        NOT NULL DEFAULT false,
    created_at    timestamptz    NOT NULL DEFAULT now()
);
CREATE INDEX ix_alerts_session_start ON alerts (session_id, start_ms);

CREATE TABLE session_summaries (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid        NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    provider    text        NOT NULL,
    model       text        NOT NULL,
    content     text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- audit
CREATE TABLE audit_logs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       uuid        REFERENCES organizations(id) ON DELETE CASCADE,
    user_id      uuid        REFERENCES users(id) ON DELETE SET NULL,
    action       text        NOT NULL,          -- e.g. login, session.delete, export, settings.update
    resource     text,
    resource_id  uuid,
    metadata     jsonb       NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_audit_logs_org_created ON audit_logs (org_id, created_at DESC);
