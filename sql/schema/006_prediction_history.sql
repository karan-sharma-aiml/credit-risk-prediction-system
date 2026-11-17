-- 006_prediction_history.sql
-- Append-only audit trail, one row per prediction: the exact inputs the model
-- saw (customer data may change later) and how the customer's risk moved since
-- the previous prediction. Deleted together with the prediction / customer.

CREATE TABLE IF NOT EXISTS credit_risk.prediction_history (
    history_id              BIGSERIAL     PRIMARY KEY,
    prediction_id           BIGINT        NOT NULL UNIQUE
                            REFERENCES credit_risk.predictions (prediction_id) ON DELETE CASCADE,
    customer_id             VARCHAR(20)   NOT NULL
                            REFERENCES credit_risk.customers (customer_id) ON DELETE CASCADE,
    input_snapshot          JSONB         NOT NULL,
    previous_prediction_id  BIGINT,
    previous_probability    NUMERIC(7, 6),
    previous_segment        VARCHAR(10),
    segment_changed         BOOLEAN       NOT NULL DEFAULT FALSE,
    probability_change      DOUBLE PRECISION,
    recorded_at             TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prediction_history_customer
    ON credit_risk.prediction_history (customer_id, recorded_at);
