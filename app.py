
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os

st.set_page_config(page_title="Insurance Estimator", page_icon="🏥", layout="wide")
st.title("🏥 Medical Insurance Premium Estimator")
st.caption("Project 03 | Applied ML — EDGE Series | DUET, Gazipur")

@st.cache_resource
def load_model():
    model  = joblib.load("model_artifacts/best_model_rf.pkl")
    scaler = joblib.load("model_artifacts/scaler.pkl")
    feats  = joblib.load("model_artifacts/feature_cols.pkl")
    return model, scaler, feats

model, scaler, FEATS = load_model()

def predict(age, sex, bmi, children, smoker, region):
    row = {f: 0 for f in FEATS}
    row["age"]      = age
    row["sex"]      = 1 if sex == "Male" else 0
    row["bmi"]      = bmi
    row["children"] = children
    row["smoker"]   = 1 if smoker == "Yes" else 0
    rc = f"region_{region.lower()}"
    if rc in row: row[rc] = 1
    X  = pd.DataFrame([row])[FEATS]
    Xs = scaler.transform(X)
    return np.expm1(model.predict(Xs)[0]), Xs

def risk(p):
    if p < 8000:  return "🟢 Low Risk"
    if p < 18000: return "🟡 Medium Risk"
    return "🔴 High Risk"

with st.sidebar:
    st.header("👤 Patient Profile")
    age      = st.slider("Age",      18, 64, 35)
    sex      = st.selectbox("Sex",   ["Male", "Female"])
    bmi      = st.slider("BMI",      10.0, 55.0, 27.0, 0.1)
    children = st.slider("Children", 0, 5, 0)
    smoker   = st.radio("Smoker?",   ["No", "Yes"], horizontal=True)
    region   = st.selectbox("Region",["Northeast","Northwest","Southeast","Southwest"])
    btn      = st.button("🔮 Estimate Premium")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("📋 Input Summary")
    st.table(pd.DataFrame({
        "Attribute": ["Age","Sex","BMI","Children","Smoker","Region"],
        "Value"    : [f"{age} yrs", sex, f"{bmi:.1f}", children, smoker, region]
    }).set_index("Attribute"))

with right:
    if btn:
        premium, Xs = predict(age, sex, bmi, children, smoker, region)
        st.success(f"### Estimated Annual Premium: **${premium:,.0f}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual",  f"${premium:,.0f}")
        c2.metric("Monthly", f"${premium/12:,.0f}")
        c3.metric("Daily",   f"${premium/365:.1f}")
        st.info(risk(premium))

        st.subheader("📊 Top-3 Cost Drivers")
        contribs = np.abs(model.feature_importances_ * Xs[0])
        feat_df  = pd.DataFrame({"Feature": FEATS, "Score": contribs})
        top3     = feat_df.nlargest(3, "Score")
        fig, ax  = plt.subplots(figsize=(5, 2.5))
        ax.barh(top3["Feature"][::-1], top3["Score"][::-1],
                color=["#e74c3c","#e67e22","#3498db"], edgecolor="white")
        ax.set_title("Top-3 Cost Drivers", fontweight="bold")
        ax.set_xlabel("Contribution Score")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("👈 Fill in the profile and click Estimate Premium")

st.divider()
st.markdown("""
> ⚠️ **Disclaimer:** Educational tool only. Not for real insurance decisions.
""")


# with open('app.py', 'w') as f:
#     f.write(app_code)
# print("✅ app.py written!")