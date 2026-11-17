from ml.config import Paths, TARGET_COL
from ml.pipeline import run_data_pipeline


def test_pipeline_end_to_end_writes_artifacts(tmp_path):
    paths = Paths.from_root(tmp_path)
    result = run_data_pipeline(paths=paths, n_rows=300, seed=3, run_eda_step=True)

    assert paths.raw_dataset.is_file()
    assert paths.clean_dataset.is_file()
    assert paths.feature_dataset.is_file()
    assert (paths.summaries / "cleaning_report.json").is_file()
    assert (paths.summaries / "feature_summary.csv").is_file()
    assert (paths.figures / "correlation_heatmap.png").is_file()

    assert result.features_df.shape[1] > result.clean_df.shape[1]
    assert result.features_df[TARGET_COL].isin([0, 1]).all()


def test_pipeline_is_reproducible(tmp_path):
    a = run_data_pipeline(paths=Paths.from_root(tmp_path / "a"), n_rows=200, seed=5, run_eda_step=False)
    b = run_data_pipeline(paths=Paths.from_root(tmp_path / "b"), n_rows=200, seed=5, run_eda_step=False)
    assert a.features_df.equals(b.features_df)
