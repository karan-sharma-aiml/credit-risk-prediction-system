"""Customer persistence (data access only; no business rules)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Customer


class CustomerRepository:
    """CRUD operations for :class:`Customer`."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, customer_id: str) -> Customer | None:
        return self.session.get(Customer, customer_id)

    def _filtered(self, search: str | None):
        stmt = select(Customer)
        if search:
            stmt = stmt.where(Customer.customer_id.ilike(f"%{search}%"))
        return stmt

    def list(self, *, limit: int = 50, offset: int = 0, search: str | None = None) -> Sequence[Customer]:
        stmt = self._filtered(search).order_by(Customer.created_at.desc(), Customer.customer_id).limit(limit).offset(offset)
        return self.session.scalars(stmt).all()

    def count(self, search: str | None = None) -> int:
        stmt = select(func.count()).select_from(self._filtered(search).subquery())
        return int(self.session.scalar(stmt) or 0)

    def create(self, data: dict[str, Any]) -> Customer:
        customer = Customer(**data)
        self.session.add(customer)
        self.session.flush()
        return customer

    def update(self, customer: Customer, data: dict[str, Any]) -> Customer:
        for key, value in data.items():
            setattr(customer, key, value)
        self.session.flush()
        return customer

    def delete(self, customer: Customer) -> None:
        self.session.delete(customer)
        self.session.flush()

    def numeric_values(self, column: str) -> list[float]:
        """Non-null values of a numeric customer column (for histograms)."""
        attr = getattr(Customer, column)
        return [float(v) for v in self.session.scalars(select(attr).where(attr.is_not(None))).all()]
