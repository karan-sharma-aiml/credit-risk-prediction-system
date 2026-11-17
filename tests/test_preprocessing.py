import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ml.features import engineer_features
from ml.pipeline import get_feature_lists
from ml.preprocessing import OutlierClipper, build_preprocessor


def test_outlier_clipper_clips_and_keeps_nan():
    train = pd.DataFrame({"x": list(range(100)) + [10_000]})
    clipper = OutlierClipper(0.0, 0.95).fit(train)
    out = clipper.transform(pd.DataFrame({"x": [5.0, 50_000.0, np.nan]}))
    assert out[0, 0] == 5.0
    assert out[1, 0] <= 100.0
    assert np.isnan(out[2, 0])


def test_outlier_clipper_requires_fit_and_valid_quantiles():
    with pytest.raises(NotFittedError):
        OutlierClipper().transform(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError):
        OutlierClipper(0.9, 0.1).fit(pd.DataFrame({"x": [1.0, 2.0]}))


def test_preprocessor_outputs_no_nan_and_handles_unseen_category(clean_df):
    features = engineer_features(clean_df)
    numeric, categorical = get_feature_lists()
    pre = build_preprocessor(numeric, categorical)
    train, new = features.iloc[:400], features.iloc[400:].copy()
    pre.fit(train)

    new.iloc[0, new.columns.get_loc("home_ownership")] = "houseboat"  # unseen level
    out = pre.transform(new)
    assert out.shape[0] == len(new)
    assert not np.isnan(out).any()
    assert len(pre.get_feature_names_out()) == out.shape[1]


def test_preprocessor_is_clonable_and_scaler_validated():
    numeric, categorical = get_feature_lists()
    clone(build_preprocessor(numeric, categorical))
    with pytest.raises(ValueError):
        build_preprocessor(numeric, categorical, scaler="minmax")
