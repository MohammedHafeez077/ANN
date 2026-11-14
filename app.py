import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ann_churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "customer_churn.csv"

st.set_page_config(page_title="NeuralPulse | Churn AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#081226 0%,#101b35 55%,#142746 100%); color:#edf4ff;}
[data-testid="stSidebar"] {background:#0c1730;}
.block-container {padding-top:2rem; max-width:1250px;}
.hero {padding:26px 30px;border:1px solid #29446c;border-radius:20px;background:linear-gradient(110deg,#14294b,#17213c);}
.hero h1 {font-size:2.25rem;margin:0;color:#f5f8ff;}
.hero p {color:#b9c9e5;margin:.5rem 0 0 0;}
.metric {background:#142542;border:1px solid #2b456a;padding:18px;border-radius:16px;}
.metric small {color:#9db3d4;}
.metric h2 {margin:5px 0;color:#f4f8ff;}
div.stButton>button {background:#4f8cff;color:white;border:0;border-radius:10px;font-weight:700;padding:.65rem 1rem;}
div.stButton>button:hover {background:#77a8ff;color:#081226;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.markdown('<div class="hero"><h1>🧠 NeuralPulse — Customer Churn Prediction</h1><p>Artificial Neural Network (ANN) • Multi-layer Perceptron • Customer retention intelligence</p></div>', unsafe_allow_html=True)
model = load_model()
if model is None:
    st.error("Trained model not found. Run: python train.py")
    st.stop()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

tab1, tab2, tab3 = st.tabs(["🔮 Predict churn", "📊 Model performance", "📁 Dataset explorer"])

with tab1:
    st.subheader("Customer profile")
    st.caption("Enter customer details and let the trained ANN estimate churn risk.")
    c1,c2,c3 = st.columns(3)
    with c1:
        age = st.slider("Age",18,85,35)
        tenure = st.slider("Tenure (months)",0,72,18)
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, max_value=25000.0, value=float(monthly*tenure), step=10.0)
    with c2:
        calls = st.slider("Support calls (last year)",0,10,1)
        contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
        internet = st.selectbox("Internet service",["Fiber","DSL","None"])
        payment = st.selectbox("Payment method",["Electronic check","Credit card","Bank transfer","Mailed check"])
    with c3:
        paperless = st.selectbox("Paperless billing",["Yes","No"])
        senior = st.selectbox("Senior citizen",["No","Yes"])
        st.info("Tip: short contracts, frequent support calls and high monthly charges can be churn signals in this demo dataset.")
    if st.button("Analyze customer", use_container_width=True):
        row = pd.DataFrame([{
            "age":age, "tenure_months":tenure, "monthly_charges":monthly,
            "total_charges":total, "support_calls":calls, "contract":contract,
            "internet_service":internet, "payment_method":payment,
            "paperless_billing":paperless, "senior_citizen":senior
        }])
        probability = float(model.predict_proba(row)[0,1])
        label = "Likely to churn" if probability >= 0.5 else "Likely to stay"
        a,b = st.columns([1,2])
        with a:
            st.metric("Prediction", label)
            st.metric("Churn probability", f"{probability:.1%}")
        with b:
            st.progress(probability)
            if probability >= 0.7:
                st.warning("High risk: consider proactive outreach, a contract review, or a tailored retention offer.")
            elif probability >= 0.4:
                st.info("Moderate risk: monitor satisfaction and service usage.")
            else:
                st.success("Low risk based on the information entered.")
        st.caption("This is an educational demonstration trained on synthetic data, not a production decision system.")

with tab2:
    st.subheader("Held-out test performance")
    cols = st.columns(5)
    for col,(name,key) in zip(cols,[("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1 score","f1"),("ROC-AUC","roc_auc")]):
        col.markdown(f'<div class="metric"><small>{name}</small><h2>{metrics[key]:.3f}</h2></div>', unsafe_allow_html=True)
    st.write("")
    st.write(f"Training records: **{metrics['train_rows']:,}** · Test records: **{metrics['test_rows']:,}**")
    st.markdown("**ANN architecture**")
    st.code("Input features → preprocessing → Dense(64, ReLU) → Dense(32, ReLU) → Sigmoid output", language="text")
    st.caption("The pipeline includes numeric standardization and one-hot encoding for categorical variables.")

with tab3:
    st.subheader("Sample dataset")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.write(f"Rows: **{len(data):,}** · Columns: **{len(data.columns)}**")
        st.dataframe(data.head(100), use_container_width=True)
        st.bar_chart(data["churn"].value_counts().rename(index={0:"Stayed",1:"Churned"}))
        st.download_button("Download CSV dataset", data.to_csv(index=False).encode("utf-8"),
                           file_name="customer_churn.csv", mime="text/csv")
    else:
        st.warning("Dataset file not found. Place customer_churn.csv inside data/.")
