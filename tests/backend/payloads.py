"""Sample API payloads."""

GOOD_CUSTOMER = {
    "age": 45, "annual_income": 95_000, "employment_type": "salaried", "employment_years": 15,
    "loan_amount": 8_000, "loan_term_months": 36, "credit_score": 780, "monthly_debt_payment": 400,
    "credit_limit": 25_000, "revolving_balance": 2_000, "savings_balance": 60_000, "monthly_expenses": 3_000,
    "num_existing_loans": 1, "num_late_payments": 0, "repayment_history": "excellent", "home_ownership": "own",
}
RISKY_CUSTOMER = {
    "age": 22, "annual_income": 18_000, "employment_type": "unemployed", "employment_years": 0.2,
    "loan_amount": 25_000, "loan_term_months": 24, "credit_score": 420, "monthly_debt_payment": 900,
    "credit_limit": 3_000, "revolving_balance": 3_300, "savings_balance": 0, "monthly_expenses": 1_400,
    "num_existing_loans": 4, "num_late_payments": 8, "repayment_history": "poor", "home_ownership": "rent",
}
