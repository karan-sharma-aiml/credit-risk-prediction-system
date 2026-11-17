"""Customer management use-cases."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from backend.app.core.exceptions import ConflictError, NotFoundError
from backend.app.database.crud import CustomerRepository
from backend.app.database.models import Customer
from backend.app.schemas.common import Page
from backend.app.schemas.customer import CustomerCreate, CustomerRead, CustomerUpdate
from backend.app.utils.ids import generate_customer_id

logger = logging.getLogger(__name__)


class CustomerService:
    """Create, read, update, delete and list customers."""

    def __init__(self, session: Session, pseudonymize: Callable[[str], str]) -> None:
        self.session = session
        self.repo = CustomerRepository(session)
        self._pseudo = pseudonymize

    def get(self, customer_id: str) -> Customer:
        customer = self.repo.get(customer_id)
        if customer is None:
            raise NotFoundError(f"Customer '{customer_id}' not found.")
        return customer

    def create(self, payload: CustomerCreate) -> Customer:
        data = payload.model_dump()
        data["customer_id"] = data["customer_id"] or generate_customer_id()
        if self.repo.get(data["customer_id"]):
            raise ConflictError(f"Customer '{data['customer_id']}' already exists.")
        customer = self.repo.create(data)
        self.session.commit()
        logger.info("customer_created", extra={"customer_ref": self._pseudo(customer.customer_id)})
        return customer

    def update(self, customer_id: str, payload: CustomerUpdate) -> Customer:
        customer = self.get(customer_id)
        self.repo.update(customer, payload.model_dump())
        self.session.commit()
        logger.info("customer_updated", extra={"customer_ref": self._pseudo(customer_id)})
        return customer

    def delete(self, customer_id: str) -> None:
        customer = self.get(customer_id)
        self.repo.delete(customer)  # predictions and history are removed by cascade
        self.session.commit()
        logger.info("customer_deleted", extra={"customer_ref": self._pseudo(customer_id)})

    def list(self, *, limit: int, offset: int, search: str | None) -> Page[CustomerRead]:
        items = self.repo.list(limit=limit, offset=offset, search=search)
        return Page[CustomerRead](
            items=[CustomerRead.model_validate(c) for c in items],
            total=self.repo.count(search),
            limit=limit,
            offset=offset,
        )

    def upsert_features(self, customer_id: str | None, features: dict[str, Any]) -> Customer:
        """Create the customer (or update its features) without committing.

        Used by the prediction flow so customer + prediction are saved atomically.
        """
        existing = self.repo.get(customer_id) if customer_id else None
        if existing is None:
            return self.repo.create({**features, "customer_id": customer_id or generate_customer_id()})
        return self.repo.update(existing, features)
