# API reference

Base URL: `http://localhost:8000` · interactive docs: `/docs` (Swagger) and `/redoc` · every response carries `X-Request-ID`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | 200 when DB reachable **and** model loaded, else 503 (used by Docker/Render/Railway health checks) |
| GET | `/model-info` | Model version, decision threshold, segment cut-offs, validation/test metrics |
| GET | `/model-info/performance` | Test-set ROC / PR curve points, confusion matrix, top feature importances (extra endpoint used by the dashboard) |
| POST | `/predict` | Validate -> score with `CreditRiskPredictor` -> save customer + prediction + audit row |
| POST | `/customers` | Create a customer (`customer_id` optional, generated if omitted) |
| GET | `/customers` | List, `limit` (1-200), `offset`, `search` (id substring) |
| GET | `/customers/{id}` | One customer |
| PUT | `/customers/{id}` | **Full replacement** (omitted optional fields become null) |
| DELETE | `/customers/{id}` | Hard delete; the customer's predictions and history are deleted by cascade |
| GET | `/predictions` | History, newest first; filters `customer_id`, `risk_segment`, `date_from`, `date_to`, paging |
| GET | `/predictions/{id}` | One prediction with its audit trail (`history`) |
| GET | `/dashboard` | KPIs, risk distribution, income/loan/credit-score histograms, 30-day trend, recent predictions |

## `POST /predict`

Required: `age` (18-100), `annual_income` (>0), `employment_type` (`salaried|self_employed|retired|unemployed`),
`loan_amount` (>0), `loan_term_months` (1-480), `credit_score` (300-850), `monthly_debt_payment` (>=0),
`savings_balance` (>=0), `monthly_expenses` (>=0). Optional (model treats as missing): `employment_years`,
`credit_limit`, `revolving_balance`, `num_existing_loans`, `num_late_payments`, `repayment_history`
(`excellent|good|fair|poor`), `home_ownership` (`rent|mortgage|own|other`), `customer_id`.
Unknown fields are rejected. `employment_years` cannot exceed `age - 14`.

If `customer_id` is omitted a customer is created; if it exists, its stored data is updated with the submitted
values before scoring (the exact inputs are kept in `prediction_history.input_snapshot`).

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "customer_id": "CUST_DEMO_1", "age": 22, "annual_income": 18000, "employment_type": "unemployed",
  "employment_years": 0.2, "loan_amount": 25000, "loan_term_months": 24, "credit_score": 420,
  "monthly_debt_payment": 900, "credit_limit": 3000, "revolving_balance": 3300, "savings_balance": 0,
  "monthly_expenses": 1400, "num_existing_loans": 4, "num_late_payments": 8,
  "repayment_history": "poor", "home_ownership": "rent"}'
```

Response (`201`, abridged: `history.input_snapshot` repeats the request):

```json
{
  "prediction_id": 1,
  "customer_id": "CUST_DEMO_1",
  "default_probability": 0.513,
  "predicted_label": 1,
  "risk_segment": "high",
  "risk_label": "High Risk",
  "recommendation": {
    "action": "reject",
    "title": "Reject Loan",
    "message": "High default risk. Decline, or offer a reduced amount with collateral.",
    "considerations": ["Debt-to-income ratio is above 43%.", "Credit utilisation is at or above 70%.",
                       "Requested loan exceeds annual income.", "Three or more late payments on record.",
                       "Savings cover less than three months of expenses."]
  },
  "model_version": "random_forest-20261115T195158Z",
  "decision_threshold": 0.08707215150548867,
  "created_at": "2026-11-16T19:30:00Z",
  "history": {"previous_prediction_id": null, "previous_segment": null, "segment_changed": false, "probability_change": null}
}
```

## Errors

All errors share one envelope; validation errors list `field`/`message`/`type` and **never echo the submitted value**.

```json
{"error": {"code": "validation_error", "message": "Request validation failed.", "request_id": "...",
           "details": [{"field": "credit_score", "message": "Input should be less than or equal to 850", "type": "less_than_equal"}]}}
```

| Status | `code` | When |
|---|---|---|
| 404 | `not_found` / `http_error` | Unknown customer, prediction or route |
| 409 | `conflict` | `POST /customers` with an existing id |
| 422 | `validation_error` | Invalid body, query or path parameter |
| 500 | `prediction_failed` / `internal_error` | Inference or unexpected failure (transaction rolled back, nothing stored) |
| 503 | `model_unavailable` | Model artifact could not be loaded at startup (`/health` also returns 503) |

## Recommendation policy (`backend/app/services/recommendation.py`)

| Segment | Action | Title |
|---|---|---|
| low | `approve` | Approve Loan |
| medium | `manual_review` | Manual Review |
| high | `reject` | Reject Loan |

Independent rules add underwriter considerations (DTI > 43%, utilisation >= 70%, loan > annual income,
>= 3 late payments, savings < 3 months of expenses). They never change the action. Both the policy and the rules are
constructor arguments of `RecommendationEngine`, so they can be replaced without touching the API.
