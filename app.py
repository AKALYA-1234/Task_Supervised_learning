# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Prediction")

# -------------------------------
# Load saved artifacts
# -------------------------------
MODEL_PATH = "artifacts/model.pkl"
PREP_PATH  = "artifacts/preprocessor.pkl"
META_PATH  = "artifacts/metadata.pkl"
TEST_PATH  = "artifacts/test_data.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREP_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

X_test_saved, y_test_saved = pickle.load(open(TEST_PATH, "rb"))

numeric_features       = metadata["numeric_features"]
categorical_features   = metadata["categorical_features"]
numeric_defaults       = metadata["numeric_defaults"]
categorical_defaults   = metadata["categorical_defaults"]
categorical_categories = metadata["categorical_categories"]
feature_order          = metadata["feature_order"]

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📦 Batch CSV", "📈 Model Evaluation"])

# -------------------------------
# Tab 1: Single Prediction
# -------------------------------
with tab1:
    st.subheader("Enter employee details")
    cols = st.columns(2)
    row = {}

    # Numeric inputs
    for i, col in enumerate(numeric_features):
        with cols[i % 2]:
            default = numeric_defaults.get(col, 0.0)
            row[col] = st.number_input(col, value=float(default))

    # Categorical inputs
    for i, col in enumerate(categorical_features):
        with cols[i % 2]:
            options = categorical_categories.get(col, [])
            default = categorical_defaults.get(col, options[0] if options else "")
            if options:
                row[col] = st.selectbox(col, options, index=options.index(default) if default in options else 0)
            else:
                row[col] = st.text_input(col, value=str(default))

    if st.button("Predict"):
        input_df = pd.DataFrame([row])[feature_order]
        X_proc = preprocessor.transform(input_df)
        proba = model.predict_proba(X_proc)[:, 1][0]
        pred = int(proba >= 0.5)

        if pred == 1:
            st.error(f"⚠️ Likely to **LEAVE** (probability: {proba*100:.2f}%)")
        else:
            st.success(f"✅ Likely to **STAY** (probability to leave: {proba*100:.2f}%)")

# -------------------------------
# Tab 2: Batch CSV Prediction
# -------------------------------
with tab2:
    st.subheader("Upload a CSV to score many employees")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df = df[feature_order]
            Xp = preprocessor.transform(df)
            probas = model.predict_proba(Xp)[:, 1]
            preds = (probas >= 0.5).astype(int)
            out = df.copy()
            out["Attrition_Prob"] = probas
            out["Attrition_Pred"] = preds

            st.success(f"Scored {len(out)} rows")
            st.dataframe(out.head(20))
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="attrition_predictions.csv",
                mime="text/csv"
            )

# -------------------------------
# Tab 3: Model Evaluation
# -------------------------------
with tab3:
    st.subheader("Confusion Matrix & ROC Curve (Test Set)")
    X_test_proc = preprocessor.transform(X_test_saved)
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test_saved, y_pred)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed","Left"], yticklabels=["Stayed","Left"], ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test_saved, y_proba)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1],[0,1], linestyle="--")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve"); ax2.legend(loc="lower right")
    st.pyplot(fig2)

    st.info(f"ROC-AUC on held-out test set: **{roc_auc:.3f}**")
