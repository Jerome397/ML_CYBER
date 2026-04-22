# ML Handoff for Backend

## Current status
The ML pipeline is ready for backend integration.

The current selected final model is stored in:

- `artifacts/final/best_model.joblib`

Supporting metadata is stored in:

- `artifacts/final/feature_order.json`
- `artifacts/final/label_mapping.json`

## Main integration file
Use:

- `ml/inference.py`

Main function:

```python
from ml.inference import predict