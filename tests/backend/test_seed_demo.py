import pytest
from sqlalchemy import create_engine, text

from backend.app.core.config import get_settings
from backend.app.scripts.seed_demo import seed
from ml.config import PATHS


@pytest.mark.skipif(not PATHS.clean_dataset.is_file(), reason="run `python -m ml.pipeline` first")
def test_seed_creates_customers_and_scored_history(tmp_path, monkeypatch):
    db = tmp_path / "seed.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()
    try:
        counts = seed(30)
        again = seed(30)  # idempotent: existing ids are skipped
    finally:
        get_settings.cache_clear()
    assert counts["created"] > 20 and counts["rescored"] >= 1
    assert again["created"] == 0 and again["skipped_existing"] == counts["created"]
    with create_engine(f"sqlite:///{db}").connect() as conn:
        n_pred = conn.execute(text("select count(*) from predictions")).scalar()
        n_hist = conn.execute(text("select count(*) from prediction_history")).scalar()
        outcomes = conn.execute(text("select count(*) from customers where is_default is not null")).scalar()
        future = conn.execute(text("select count(*) from predictions where predicted_at > datetime('now')")).scalar()
    assert future == 0
    assert n_pred == n_hist == counts["created"] + counts["rescored"] and outcomes == counts["created"]
