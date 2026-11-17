-- 004_model_metrics.sql
-- Long-format evaluation metrics (one row per model / split / metric), which
-- stays stable as new metrics are added in later parts.

CREATE TABLE IF NOT EXISTS credit_risk.model_metrics (
    metric_id      BIGSERIAL        PRIMARY KEY,
    model_id       BIGINT           NOT NULL
                   REFERENCES credit_risk.model_registry (model_id) ON DELETE CASCADE,
    dataset_split  VARCHAR(12)      NOT NULL
                   CHECK (dataset_split IN ('train', 'validation', 'test')),
    metric_name    VARCHAR(50)      NOT NULL,    -- roc_auc, pr_auc, recall, ks, ...
    metric_value   DOUBLE PRECISION NOT NULL,
    evaluated_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),
    UNIQUE (model_id, dataset_split, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON credit_risk.model_metrics (metric_name);
