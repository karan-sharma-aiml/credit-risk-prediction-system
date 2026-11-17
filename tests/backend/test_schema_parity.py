"""The ORM models and sql/schema/*.sql must describe the same tables and columns."""

import re
from pathlib import Path

from backend.app.database import models
from backend.app.database.base import Base
from ml.config import PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql" / "schema"
SQL_ONLY_TABLES = {"feature_summaries"}


def _parse_sql_tables() -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for path in sorted(Path(SQL_DIR).glob("*.sql")):
        text = re.sub(r"--[^\n]*", "", path.read_text())
        for match in re.finditer(r"CREATE TABLE IF NOT EXISTS credit_risk\.(\w+)\s*\((.*?)\n\);", text, re.S):
            columns = set()
            for line in match.group(2).split("\n"):
                token = line.strip().split(" ")[0].strip(",")
                if re.fullmatch(r"[a-z_0-9]+", token) and token not in {"unique", "check", "primary", "foreign", "constraint"}:
                    columns.add(token)
            tables[match.group(1)] = columns
    return tables


def test_every_orm_table_exists_in_sql_with_identical_columns():
    assert models.Base is Base
    sql = _parse_sql_tables()
    orm = {t.name: {c.name for c in t.columns} for t in Base.metadata.tables.values()}
    assert set(orm) <= set(sql), f"ORM tables missing in SQL: {set(orm) - set(sql)}"
    assert set(sql) - set(orm) == SQL_ONLY_TABLES
    for table, columns in orm.items():
        assert columns == sql[table], f"{table}: ORM-only {columns - sql[table]}, SQL-only {sql[table] - columns}"
