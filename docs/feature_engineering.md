# Feature engineering

All features live in `ml/features.py`, are **stateless** (a function of one row) and
return `NaN` when an input is missing or a denominator is zero. Ratios are capped
(see `FeatureSettings` in `ml/config.py`) so one tiny denominator cannot create
absurd values.

| Feature | Formula | Why it matters in credit-risk modelling |
|---|---|---|
| `debt_to_income` | `monthly_debt_payment / (annual_income / 12)` | Core affordability metric; underwriting rules of thumb flag ~36-43%+. |
| `loan_to_income` | `loan_amount / annual_income` | Size of the new obligation relative to earning power, independent of existing debt. |
| `loan_payment_to_income` | `(loan_amount / loan_term_months) / monthly_income` | Principal-only proxy for the new monthly burden (interest ignored); the only feature that uses loan term. |
| `expense_ratio` | `monthly_expenses / monthly_income` | Little disposable income means little capacity to absorb a repayment. |
| `savings_ratio` | `savings_balance / annual_income` | Financial buffer; savings reduce default risk when income is disrupted. |
| `savings_runway_months` | `savings_balance / monthly_expenses` | Months of expenses covered without income. |
| `credit_utilization` | `revolving_balance / credit_limit` | Bureau-style stress signal; high utilisation strongly predicts default. |
| `high_utilization_flag` | `1 if utilization >= 0.70` | Non-linear risk tier that simple models can use directly. |
| `employment_stability_score` | `type_weight x min(years / 10, 1)` | Proxy for income continuity. Weights (salaried 1.0, retired 0.9, self-employed 0.7, unemployed 0) are **heuristic assumptions**, tunable in config. |
| `has_late_payments` | `1 if num_late_payments > 0` | Past delinquency is one of the strongest default predictors. |
| `age_group` | bins 18-24, 25-34, 35-44, 45-54, 55-64, 65+ | Life-stage effects (e.g. thin-file young borrowers). Review for fairness before production use. |
| `income_bucket` | bins low / lower_mid / mid / upper_mid / high (edges 25k, 50k, 80k, 120k) | Non-linear income effects and a ready-made segmentation axis. |

## Design decisions

- **Fixed bucket edges, not quantiles.** Quantile bins would be fit on data
  (leakage risk, and a customer's bucket could change between runs). Fixed edges
  give the same answer in training, batch scoring and the API.
- **Missing stays missing.** No feature silently converts missing to 0; the
  imputer fitted on the training split decides, and missing-indicators are added.
- **Fairness note.** `age_group` is a legitimate risk signal in some
  jurisdictions and a regulated attribute in others. Confirm with compliance
  before using it in a production decision.
