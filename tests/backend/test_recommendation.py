import pytest

from backend.app.services.recommendation import DEFAULT_POLICY, RecommendationEngine, RiskRule, derive_features
from tests.backend.payloads import GOOD_CUSTOMER, RISKY_CUSTOMER


@pytest.mark.parametrize(("segment", "action"), [("low", "approve"), ("medium", "manual_review"), ("high", "reject")])
def test_policy_maps_segment_to_action(segment, action):
    assert RecommendationEngine().recommend(segment).action == action


def test_titles_match_business_wording():
    engine = RecommendationEngine()
    assert [engine.recommend(s).title for s in ("low", "medium", "high")] == ["Approve Loan", "Manual Review", "Reject Loan"]


def test_rules_fire_on_risky_profile_only():
    engine = RecommendationEngine()
    assert engine.recommend("low", derive_features(GOOD_CUSTOMER)).considerations == []
    notes = engine.recommend("high", derive_features(RISKY_CUSTOMER)).considerations
    assert any("Debt-to-income" in n for n in notes) and any("late payments" in n for n in notes)
    assert any("utilisation" in n for n in notes)


def test_missing_features_do_not_trigger_rules():
    assert RecommendationEngine().considerations({"debt_to_income": None}) == []
    assert RecommendationEngine().considerations(None) == []


def test_engine_is_extensible_with_custom_policy_and_rules():
    rule = RiskRule("always", lambda f: True, "custom note")
    engine = RecommendationEngine({"low": ("fast_track", "Fast Track", "x")}, [rule])
    rec = engine.recommend("low", {"anything": 1.0})
    assert rec.action == "fast_track" and rec.considerations == ["custom note"]


def test_from_stored_roundtrip_and_unknown_action():
    engine = RecommendationEngine()
    stored = engine.from_stored("reject", ["note"])
    assert stored.title == "Reject Loan" and stored.considerations == ["note"]
    assert engine.from_stored("legacy_action", None).title == "Legacy Action"
    assert set(DEFAULT_POLICY) == {"low", "medium", "high"}


def test_derive_features_reuses_ml_feature_engineering():
    f = derive_features(GOOD_CUSTOMER)
    assert f["debt_to_income"] == pytest.approx(400 / (95_000 / 12))
    assert f["credit_utilization"] == pytest.approx(2_000 / 25_000)
