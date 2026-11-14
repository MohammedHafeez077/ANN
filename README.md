# NeuralPulse — ANN Customer Churn Prediction

A complete beginner-friendly Artificial Neural Network (ANN) project with a Streamlit dashboard. It predicts whether a customer is likely to leave a subscription service.

## Project contents

- `app.py` — Streamlit UI with prediction, performance and dataset tabs
- `train.py` — trains and evaluates the ANN pipeline
- `data/customer_churn.csv` — 5,000-row synthetic demonstration dataset
- `models/ann_churn_pipeline.joblib` — pre-trained model pipeline
- `models/metrics.json` — held-out test metrics
- `requirements.txt` — Python dependencies

## Important note about the data

The included dataset is synthetic and generated for learning and demonstration. It is not real customer data and model scores should not be treated as real-world business performance.

## 1. Install Python

Use Python 3.10–3.12 for the smoothest compatibility with the listed packages. Open a terminal in this project folder.

## 2. Create and activate a virtual environment

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the app
```bash
streamlit run app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`.

## 5. Retrain the ANN (optional)
```bash
python train.py
```
This recreates the saved model and metrics from `data/customer_churn.csv`.

## ANN workflow

1. Load customer records and separate features from the `churn` target.
2. Standardize numeric columns using `StandardScaler`.
3. One-hot encode categorical columns.
4. Feed processed features into an MLP neural network.
5. The network uses two hidden layers: 64 and 32 neurons, with ReLU activation.
6. The output estimates churn probability; values at or above 0.50 are labeled likely to churn.
7. Evaluate on a held-out test set using accuracy, precision, recall, F1 and ROC-AUC.

## Troubleshooting

- **`ModuleNotFoundError`**: activate the virtual environment and run `pip install -r requirements.txt`.
- **Model file not found**: run `python train.py` from the project folder. The script uses paths relative to itself.
- **Wrong folder in terminal**: run `cd path\to\ann_customer_churn_project` before installing or launching.
- **Streamlit command not found**: use `python -m streamlit run app.py`.
- **Python version issues**: use Python 3.11 in a fresh virtual environment.

## Suggested viva explanation

"This project uses an Artificial Neural Network, specifically a Multi-Layer Perceptron, to predict customer churn. We preprocess numeric and categorical data, train the neural network on 80% of the records, and evaluate it on the remaining 20%. The Streamlit application allows a user to enter customer details and view the predicted churn probability."
