# Notebooks

Notebooks are for **exploration only**. Any logic worth keeping must live in
`ml/` as a tested function and be imported here, e.g.:

```python
from ml.pipeline import run_data_pipeline
result = run_data_pipeline()
result.features_df.head()
```

Naming convention: `NN_short_description.ipynb` (e.g. `01_eda_review.ipynb`).
