-- 001_customers.sql
-- Customer master data: one row per customer, mirroring the cleaned dataset
-- produced by ml/preprocessing/cleaning.py. Columns are nullable where the
-- pipeline deliberately keeps missing values (imputed later inside the model
-- pipeline, never silently in storage).

CREATE TABLE IF NOT EXISTS credit_risk.customers (
    customer_id           VARCHAR(20)    PRIMARY KEY,
    age                   SMALLINT       CHECK (age BETWEEN 18 AND 100),
    annual_income         NUMERIC(14, 2) CHECK (annual_income > 0),
    employment_type       VARCHAR(20)    CHECK (employment_type IN
                                          ('salaried', 'self_employed', 'retired', 'unemployed')),
    employment_years      NUMERIC(4, 1)  CHECK (employment_years BETWEEN 0 AND 60),
    loan_amount           NUMERIC(14, 2) CHECK (loan_amount > 0),
    loan_term_months      SMALLINT       CHECK (loan_term_months BETWEEN 1 AND 480),
    credit_score          SMALLINT       CHECK (credit_score BETWEEN 300 AND 850),
    monthly_debt_payment  NUMERIC(12, 2) CHECK (monthly_debt_payment >= 0),
    credit_limit          NUMERIC(14, 2) CHECK (credit_limit >= 0),
    revolving_balance     NUMERIC(14, 2) CHECK (revolving_balance >= 0),
    savings_balance       NUMERIC(14, 2) CHECK (savings_balance >= 0),
    monthly_expenses      NUMERIC(12, 2) CHECK (monthly_expenses >= 0),
    num_existing_loans    SMALLINT       CHECK (num_existing_loans >= 0),
    num_late_payments     SMALLINT       CHECK (num_late_payments >= 0),
    repayment_history     VARCHAR(10)    CHECK (repayment_history IN
                                          ('excellent', 'good', 'fair', 'poor')),
    home_ownership        VARCHAR(10)    CHECK (home_ownership IN
                                          ('rent', 'mortgage', 'own', 'other')),
    -- Observed outcome (dataset column "default"; renamed because DEFAULT is
    -- a reserved SQL keyword). NULL for new applicants with no outcome yet.
    is_default            BOOLEAN,
    source_dataset        VARCHAR(100)   NOT NULL,
    created_at            TIMESTAMPTZ    NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ    NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_customers_is_default ON credit_risk.customers (is_default);
CREATE INDEX IF NOT EXISTS idx_customers_source ON credit_risk.customers (source_dataset);
