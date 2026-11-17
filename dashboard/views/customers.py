"""Customer management page: browse, create, edit, delete."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import APIClient, APIError
from dashboard.forms import customer_form
from dashboard.ui import show_api_error

PAGE_SIZE = 25


def _browse(client: APIClient) -> None:
    search = st.text_input("Search by customer id", key="cust_search")
    page_no = st.number_input("Page", min_value=1, value=1, step=1, key="cust_page")
    data = client.list_customers(limit=PAGE_SIZE, offset=(int(page_no) - 1) * PAGE_SIZE, search=search or None)
    st.caption(f"{data['total']:,} customers · page {int(page_no)}")
    if data["items"]:
        st.dataframe(pd.DataFrame(data["items"]), width="stretch", hide_index=True)
    else:
        st.info("No customers found.")


def _create(client: APIClient) -> None:
    payload = customer_form("create_customer", submit_label="Create customer", include_outcome=True)
    if payload is None:
        return
    try:
        created = client.create_customer(payload)
        st.success(f"Customer **{created['customer_id']}** created.")
    except APIError as exc:
        show_api_error(exc)


def _edit_delete(client: APIClient) -> None:
    customer_id = st.text_input("Customer id", key="edit_id").strip()
    if not customer_id:
        return
    try:
        current = client.get_customer(customer_id)
    except APIError as exc:
        show_api_error(exc)
        return
    payload = customer_form(f"edit_{customer_id}", current, submit_label="Save changes", include_outcome=True)
    if payload is not None:
        try:
            client.update_customer(customer_id, payload)
            st.success("Customer updated.")
        except APIError as exc:
            show_api_error(exc)
    st.divider()
    confirm = st.checkbox("I understand this also deletes the customer's predictions", key=f"confirm_{customer_id}")
    if st.button("Delete customer", disabled=not confirm, type="secondary"):
        try:
            client.delete_customer(customer_id)
            st.success("Customer deleted.")
        except APIError as exc:
            show_api_error(exc)


def render(client: APIClient) -> None:
    st.title("Customer Management")
    browse, create, edit = st.tabs(["Browse", "Create", "Edit / Delete"])
    with browse:
        _browse(client)
    with create:
        _create(client)
    with edit:
        _edit_delete(client)
