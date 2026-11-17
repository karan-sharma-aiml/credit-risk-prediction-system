"""Customer management endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status

from backend.app.dependencies import CustomerServiceDep
from backend.app.schemas.common import Page
from backend.app.schemas.customer import CUSTOMER_ID_PATTERN, CustomerCreate, CustomerRead, CustomerUpdate

router = APIRouter(prefix="/customers", tags=["customers"])
CustomerId = Annotated[str, Path(pattern=CUSTOMER_ID_PATTERN)]


@router.post("", response_model=CustomerRead, status_code=status.HTTP_201_CREATED, summary="Create a customer")
def create_customer(payload: CustomerCreate, service: CustomerServiceDep) -> CustomerRead:
    return CustomerRead.model_validate(service.create(payload))


@router.get("", response_model=Page[CustomerRead], summary="List customers (paginated, optional id search)")
def list_customers(
    service: CustomerServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    search: Annotated[str | None, Query(max_length=20)] = None,
) -> Page[CustomerRead]:
    return service.list(limit=limit, offset=offset, search=search)


@router.get("/{customer_id}", response_model=CustomerRead, summary="Get one customer")
def get_customer(customer_id: CustomerId, service: CustomerServiceDep) -> CustomerRead:
    return CustomerRead.model_validate(service.get(customer_id))


@router.put("/{customer_id}", response_model=CustomerRead, summary="Replace a customer's data")
def update_customer(customer_id: CustomerId, payload: CustomerUpdate, service: CustomerServiceDep) -> CustomerRead:
    return CustomerRead.model_validate(service.update(customer_id, payload))


@router.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a customer and their predictions")
def delete_customer(customer_id: CustomerId, service: CustomerServiceDep) -> Response:
    service.delete(customer_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
