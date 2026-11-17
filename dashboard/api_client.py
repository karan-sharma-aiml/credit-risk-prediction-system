"""Thin HTTP client for the FastAPI backend (the dashboard never touches the database)."""

from __future__ import annotations

from typing import Any

import requests

from dashboard.config import api_url


class APIError(Exception):
    """Raised for connection problems and non-2xx responses."""

    def __init__(self, message: str, status_code: int | None = None, details: list[dict] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or []


class APIClient:
    """Typed-ish wrapper around the REST API."""

    def __init__(self, base_url: str, timeout: float = 20.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the API at {self.base_url}.") from exc
        if response.status_code == 204:
            return None
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not response.ok:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            raise APIError(
                error.get("message", f"API returned HTTP {response.status_code}."),
                response.status_code,
                error.get("details"),
            )
        return body

    # health / model
    def health(self) -> dict:
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        return self._request("GET", "/model-info")

    def model_performance(self) -> dict:
        return self._request("GET", "/model-info/performance")

    def dashboard(self) -> dict:
        return self._request("GET", "/dashboard")

    # customers
    def list_customers(self, limit: int = 50, offset: int = 0, search: str | None = None) -> dict:
        params = {"limit": limit, "offset": offset, **({"search": search} if search else {})}
        return self._request("GET", "/customers", params=params)

    def get_customer(self, customer_id: str) -> dict:
        return self._request("GET", f"/customers/{customer_id}")

    def create_customer(self, payload: dict) -> dict:
        return self._request("POST", "/customers", json=payload)

    def update_customer(self, customer_id: str, payload: dict) -> dict:
        return self._request("PUT", f"/customers/{customer_id}", json=payload)

    def delete_customer(self, customer_id: str) -> None:
        self._request("DELETE", f"/customers/{customer_id}")

    # predictions
    def predict(self, payload: dict) -> dict:
        return self._request("POST", "/predict", json=payload)

    def list_predictions(self, limit: int = 50, offset: int = 0, **filters: Any) -> dict:
        params = {"limit": limit, "offset": offset, **{k: v for k, v in filters.items() if v}}
        return self._request("GET", "/predictions", params=params)

    def get_prediction(self, prediction_id: int) -> dict:
        return self._request("GET", f"/predictions/{prediction_id}")


def get_client() -> APIClient:
    """Client for the configured ``API_URL``."""
    return APIClient(api_url())
