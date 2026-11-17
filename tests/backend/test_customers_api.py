import pytest

from tests.backend.payloads import GOOD_CUSTOMER


def _create(client, **overrides):
    return client.post("/customers", json={**GOOD_CUSTOMER, **overrides})


def test_customer_crud_lifecycle(client):
    created = _create(client, customer_id="CUST_API_1", source_dataset="unit-test")
    assert created.status_code == 201
    body = created.json()
    assert body["customer_id"] == "CUST_API_1" and body["annual_income"] == 95_000

    assert client.get("/customers/CUST_API_1").json()["credit_score"] == 780

    update = {**GOOD_CUSTOMER, "credit_score": 700, "is_default": False}
    assert client.put("/customers/CUST_API_1", json=update).json()["credit_score"] == 700

    assert client.delete("/customers/CUST_API_1").status_code == 204
    assert client.get("/customers/CUST_API_1").status_code == 404


def test_create_generates_id_and_rejects_duplicates(client):
    generated = _create(client).json()["customer_id"]
    assert generated.startswith("CUST") and len(generated) <= 20
    conflict = _create(client, customer_id=generated)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "conflict"


def test_put_is_full_replacement(client):
    _create(client, customer_id="CUST_PUT")
    minimal = {k: v for k, v in GOOD_CUSTOMER.items() if k not in ("credit_limit", "home_ownership")}
    replaced = client.put("/customers/CUST_PUT", json=minimal).json()
    assert replaced["credit_limit"] is None and replaced["home_ownership"] is None


def test_not_found_and_invalid_id(client):
    assert client.get("/customers/MISSING1").status_code == 404
    assert client.put("/customers/MISSING1", json=GOOD_CUSTOMER).status_code == 404
    assert client.delete("/customers/MISSING1").status_code == 404
    assert client.get("/customers/bad id!").status_code == 422


def test_list_pagination_and_search(client):
    for i in range(5):
        _create(client, customer_id=f"CUST_L{i}")
    _create(client, customer_id="OTHER_1")
    page = client.get("/customers", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 6 and len(page["items"]) == 2 and page["limit"] == 2
    found = client.get("/customers", params={"search": "cust_l"}).json()
    assert found["total"] == 5
    assert client.get("/customers", params={"limit": 1000}).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 17), ("age", 101), ("annual_income", 0), ("annual_income", -5), ("loan_amount", 0),
        ("loan_term_months", 0), ("loan_term_months", 481), ("credit_score", 299), ("credit_score", 851),
        ("savings_balance", -1), ("monthly_expenses", -1), ("monthly_debt_payment", -1),
        ("employment_type", "astronaut"), ("repayment_history", "terrible"), ("home_ownership", "castle"),
        ("num_late_payments", -1), ("age", "forty"),
    ],
)
def test_validation_rejects_bad_values_without_echoing_input(client, field, value):
    response = _create(client, **{field: value})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any(d["field"] == field for d in error["details"])
    if isinstance(value, str):
        assert value not in response.text  # offending input is not echoed back
    assert "input" not in {k for d in error["details"] for k in d}


def test_validation_requires_core_fields_and_forbids_unknown(client):
    missing = {k: v for k, v in GOOD_CUSTOMER.items() if k != "credit_score"}
    assert client.post("/customers", json=missing).status_code == 422
    assert _create(client, surprise="x").status_code == 422


def test_employment_years_cannot_exceed_working_age(client):
    response = _create(client, age=20, employment_years=10)
    assert response.status_code == 422
    assert "employment_years" in response.text
