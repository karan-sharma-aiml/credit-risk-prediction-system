-- 005_feature_summaries.sql
-- Per-feature distribution statistics per dataset version. Columns match
-- reports/summaries/feature_summary.csv (written by ml/eda.py), so the CSV
-- can be bulk-loaded with COPY. Also the baseline for later drift monitoring.

CREATE TABLE IF NOT EXISTS credit_risk.feature_summaries (
    summary_id       BIGSERIAL        PRIMARY KEY,
    dataset_version  VARCHAR(50)      NOT NULL,
    feature_name     VARCHAR(100)     NOT NULL,
    feature_type     VARCHAR(12)      NOT NULL CHECK (feature_type IN ('numeric', 'categorical')),
    n_obs            INTEGER          NOT NULL,
    n_missing        INTEGER          NOT NULL,
    mean             DOUBLE PRECISION,
    std              DOUBLE PRECISION,
    min_value        DOUBLE PRECISION,
    p25              DOUBLE PRECISION,
    median           DOUBLE PRECISION,
    p75              DOUBLE PRECISION,
    max_value        DOUBLE PRECISION,
    n_unique         INTEGER,
    generated_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    UNIQUE (dataset_version, feature_name)
);
