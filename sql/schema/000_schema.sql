-- 000_schema.sql
-- Namespace for every credit-risk object. Target engine: PostgreSQL 14+.
-- Apply files in numeric order (see sql/README.md).

CREATE SCHEMA IF NOT EXISTS credit_risk;
SET search_path TO credit_risk, public;
