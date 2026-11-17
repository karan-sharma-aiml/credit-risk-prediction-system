-- 003_predictions.sql
-- Model outputs per customer: default probability, label and risk segment.
-- Append-only: re-scoring inserts a new row, preserving history for audit.

CREATE TABLE IF NOT EXISTS credit_risk.predictions (
    prediction_id        BIGSERIAL     PRIMARY KEY,
    customer_id          VARCHAR(20)   NOT NULL
                         REFERENCES credit_risk.customers (customer_id) ON DELETE CASCADE,
    model_id             BIGINT        NOT NULL
                         REFERENCES credit_risk.model_registry (model_id),
    default_probability  NUMERIC(7, 6) NOT NULL CHECK (default_probability BETWEEN 0 AND 1),
    predicted_label      SMALLINT      NOT NULL CHECK (predicted_label IN (0, 1)),
    decision_threshold   NUMERIC(5, 4) CHECK (decision_threshold BETWEEN 0 AND 1),
    risk_segment         VARCHAR(10)   NOT NULL
                         CHECK (risk_segment IN ('low', 'medium', 'high')),
    recommendation_action VARCHAR(20),  -- approve / manual_review / reject
    top_risk_factors     JSONB,        -- rule-based considerations (recommendation engine)
    request_id           VARCHAR(64),  -- correlates with API logs (X-Request-ID)
    predicted_at         TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer_time
    ON credit_risk.predictions (customer_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON credit_risk.predictions (model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_segment ON credit_risk.predictions (risk_segment);
