"""Load demo customers and score them through the real service layer.

    python -m backend.app.scripts.seed_demo --n 300

Reads the cleaned dataset produced by ``python -m ml.pipeline`` (``data/interim``),
creates customers (skipping ids that already exist) and scores each one with the
saved model, spreading timestamps over the last 30 days so the dashboard trend has
data. Every 10th customer is re-scored with a lower credit score to exercise the
prediction history (risk migration). Uses the exact same code path as the API.
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import pandas as pd
from pydantic import ValidationError

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, pseudonymize
from backend.app.database.models import utcnow
from backend.app.database.session import create_db_engine, create_session_factory, init_db, session_scope
from backend.app.schemas.customer import CustomerBase, CustomerCreate
from backend.app.schemas.prediction import PredictionRequest
from backend.app.services.model_service import ModelContext, register_active_model
from backend.app.services.prediction_service import PredictionService
from backend.app.services.recommendation import RecommendationEngine
from ml.config import PATHS, TARGET_COL
from ml.predict import CreditRiskPredictor

logger = logging.getLogger("seed_demo")
FEATURES = tuple(CustomerBase.model_fields)


def _row_to_features(row: pd.Series) -> dict:
    """Dataset row -> feature dict with NaN replaced by ``None``."""
    return {k: (None if pd.isna(row[k]) else row[k].item() if hasattr(row[k], "item") else row[k]) for k in FEATURES}


def seed(n: int, source: str = "synthetic-seed") -> dict[str, int]:
    """Create and score up to ``n`` customers; returns counts of created / skipped rows."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    data = pd.read_csv(PATHS.clean_dataset).head(n)
    engine = create_db_engine(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    predictor = CreditRiskPredictor.from_path(settings.model_path)
    secret = settings.secret_key.get_secret_value()
    counts = {"created": 0, "skipped_invalid": 0, "skipped_existing": 0, "rescored": 0}

    with session_scope(factory) as session:
        model_id = register_active_model(session, predictor, settings.model_path)

    now = utcnow()
    for i, (_, row) in enumerate(data.iterrows()):
        with session_scope(factory) as session:
            service = PredictionService(
                session, ModelContext(predictor, model_id), RecommendationEngine(), lambda v: pseudonymize(v, secret)
            )
            features = _row_to_features(row)
            try:
                create = CustomerCreate(
                    **features, customer_id=row["customer_id"], is_default=bool(row[TARGET_COL]), source_dataset=source
                )
            except ValidationError:
                counts["skipped_invalid"] += 1
                continue
            if service.customer_repo.get(create.customer_id):
                counts["skipped_existing"] += 1
                continue
            service.customers.create(create)
            when = now - timedelta(days=i % 30, hours=(i * 7) % 24)
            service.predict(PredictionRequest(**features, customer_id=create.customer_id), created_at=when)
            counts["created"] += 1

            if i % 10 == 0:  # a later re-score with a worse credit score -> history shows migration
                worse = {**features, "credit_score": max(300, int(features["credit_score"]) - 80)}
                service.predict(
                    PredictionRequest(**worse, customer_id=create.customer_id), created_at=min(when + timedelta(hours=6), now)  # never in the future
                )
                counts["rescored"] += 1
    engine.dispose()
    logger.info("seed_finished", extra=counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo customers and predictions.")
    parser.add_argument("--n", type=int, default=300, help="Number of dataset rows to load")
    args = parser.parse_args()
    print(seed(args.n))


if __name__ == "__main__":
    main()
