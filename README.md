# Cybersecurity ML Streaming Demo

## Overview

This project builds a complete machine learning pipeline for cybersecurity traffic classification.
It includes:

* Training multiple models (Logistic Regression, Random Forest, Neural Network)
* Hyperparameter tuning
* Model selection and promotion
* Real-time (manual-triggered) prediction simulation
* Interactive visualization using Streamlit

The system supports three datasets:

* Binary classification (Attack vs Benign)
* 7-class classification
* 33-class classification

---

## Project Structure

```
Project-ML-cyber/
│
├── ml/                         # Training and tuning scripts
├── demo_app/                  # Backend + Streamlit frontend
├── artifacts/                 # All model outputs
│   ├── runs/                 # Individual training runs
│   ├── final_binary/
│   ├── final_multiclass/
│   ├── final_multiclass_nn/
│   ├── final_33class/
│
├── dataset/                   # Datasets
├── .gitignore
└── README.md
```

---

## Important Note About Artifacts

The `artifacts/` folder is **excluded from GitHub** because it contains large trained models.

To regenerate everything:

* Run training scripts
* Run promotion scripts

---

## Models Used

### Binary Dataset

* Logistic Regression
* Random Forest

### 7-Class Dataset

* Logistic Regression
* Random Forest
* Neural Network (MLP)

### 33-Class Dataset

* Logistic Regression
* Random Forest

---

## Training

### Binary Models

```bash
py ml\train_logistic_regression_binary.py
py ml\train_random_forest_binary.py
```

### Multiclass Models

```bash
py ml\train_logistic_regression_multiclass.py
py ml\train_random_forest_multiclass.py
py ml\train_mlp_multiclass.py
```

### 33-Class Models

```bash
py ml\train_logistic_regression_33class.py
py ml\train_random_forest_33class.py
```

---

## Hyperparameter Tuning

Neural network tuning:

```bash
py ml\tune_mlp_multiclass.py
```

This creates multiple runs stored in:

```
artifacts/runs/mlp_tuned_...
```

Each run keeps:

* parameters used
* validation scores
* test scores

---

## Model Promotion

After training, select the best model for each dataset.

### Binary

```bash
py ml\promote_best_binary_model.py
```

### Multiclass

```bash
py ml\promote_best_multiclass_model.py
```

### 33-Class

```bash
py ml\promote_best_33class_model.py
```

These scripts:

* compare models using **macro F1-score**
* copy the best model into `artifacts/final_*`

---

## Backend

The backend is built using FastAPI.

### Start backend

```bash
py -m uvicorn demo_app.backend:app --reload
```

### Available endpoints

* `/health` → check system status
* `/tick` → process one data row and predict
* `/state` → get prediction logs
* `/reset` → reset logs
* `/dashboard` → get model metrics for visualization

---

## Frontend (Streamlit)

### Run Streamlit

```bash
py -m streamlit run demo_app/streamlit_app.py
```

---

## Demo Functionality

### Page 1 — Model Comparison

* Shows metrics for each model
* Displays:

  * Accuracy
  * Macro F1-score
* Explains why a model was selected

### Page 2 — Manual Prediction Logs

* Button: **Process Next Line**
* Each click:

  * Takes one row from dataset
  * Transforms it into features
  * Sends it to model
  * Displays:

    * prediction
    * actual label
    * correctness
    * confidence

Tables available for:

* Binary model
* Multiclass best model
* Multiclass neural network
* 33-class model

---

## How Prediction Works

1. A row is read from test dataset
2. Features are extracted using `feature_order.json`
3. Model predicts label
4. Label is mapped using `label_mapping.json`
5. Result is stored and displayed

---

## Technologies Used

* Python
* Scikit-learn
* Pandas / NumPy
* FastAPI
* Streamlit
* Plotly

---

## Notes

* Macro F1-score is prioritized due to class imbalance
* Neural network is used only for multiclass dataset
* All models use the same feature ordering for consistency

---

## Future Improvements

* Real-time streaming instead of manual trigger
* Deployment (Docker / cloud)
* Model monitoring dashboard
* Dataset expansion

---

## Author

Jerome Khater

Joseph Yaghi

Karl Iskandar

Lynn Najm
Computer & Communication Engineering (AI Branch)
