-- 002_model_registry.sql
-- Supporting table (beyond the four required) so predictions and metrics can
-- reference the exact model version that produced them.

CREATE TABLE IF NOT EXISTS credit_risk.model_registry (
    model_id         BIGSERIAL    PRIMARY KEY,
    model_name       VARCHAR(100) NOT NULL,
    model_version    VARCHAR(50)  NOT NULL,
    algorithm        VARCHAR(100),
    artifact_path    TEXT,                       -- e.g. saved_models/xgb_v1.joblib
    dataset_version  VARCHAR(50),
    hyperparameters  JSONB,
    is_active        BOOLEAN      NOT NULL DEFAULT FALSE,
    trained_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (model_name, model_version)
);

-- At most one active model per model_name.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_one_active
    ON credit_risk.model_registry (model_name) WHERE is_active;
