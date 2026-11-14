from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
target = "churn"
X, y = df.drop(columns=[target]), df[target]
numeric = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
categorical = ["contract", "internet_service", "payment_method", "paperless_billing", "senior_citizen"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                    alpha=0.001, learning_rate_init=0.001, max_iter=350,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=20, random_state=42)
pipeline = Pipeline([("preprocessor",preprocessor),("ann",ann)])

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:,1]
metrics = {
    "accuracy":float(accuracy_score(y_test,pred)),
    "precision":float(precision_score(y_test,pred,zero_division=0)),
    "recall":float(recall_score(y_test,pred,zero_division=0)),
    "f1":float(f1_score(y_test,pred,zero_division=0)),
    "roc_auc":float(roc_auc_score(y_test,prob)),
    "train_rows":int(len(y_train)),
    "test_rows":int(len(y_test))
}
joblib.dump(pipeline, MODEL_DIR/"ann_churn_pipeline.joblib")
with open(MODEL_DIR/"metrics.json","w") as f:
    json.dump(metrics,f,indent=2)
print("Training complete.")
print(json.dumps(metrics,indent=2))
