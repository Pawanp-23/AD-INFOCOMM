-- Development seed data. Never use in staging/production.
-- Users are created without a usable password; set one via the backend CLI once it exists.

INSERT INTO organizations (id, name)
VALUES ('00000000-0000-0000-0000-000000000001', 'AD-INFOCOMM Dev');

INSERT INTO users (org_id, email, full_name, password_hash, role)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'admin@ramei.local',   'Dev Admin',   '!disabled', 'admin'),
    ('00000000-0000-0000-0000-000000000001', 'analyst@ramei.local', 'Dev Analyst', '!disabled', 'analyst'),
    ('00000000-0000-0000-0000-000000000001', 'agent@ramei.local',   'Dev Agent',   '!disabled', 'agent');
