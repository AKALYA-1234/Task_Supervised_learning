# trained_model.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

def load_and_clean(path):
    df = pd.read_csv(path)
    # Fix odd BOM in column names if present
    df.columns = df.columns.str.replace("ï»¿", "", regex=False).str.strip()

    # Target: map Yes/No -> 1/0
    if df["Attrition"].dtype == "object":
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Drop clearly useless/leaky identifiers
    drop_cols = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df

def main():
    df = load_and_clean(DATA_PATH)

    # Split features/target
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].astype(int)

    # Identify dtypes
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessor: scale numeric, one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit preprocessor on train only
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    # SMOTE on the preprocessed training data (NOT on test!)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_p, y_train)

    # XGBoost (tuned for solid baseline)
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X_train_bal, y_train_bal)

    # Evaluate
    y_pred = model.predict(X_test_p)
    y_proba = model.predict_proba(X_test_p)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    print("\n✅ Model trained")
    print(f"Accuracy      : {acc:.3f}")
    print(f"Precision     : {prec:.3f}")
    print(f"Recall        : {rec:.3f}")
    print(f"F1-score      : {f1:.3f}")
    print(f"ROC-AUC       : {auc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Attach helpful metadata to the preprocessor
    # (so the app can build a form with sensible defaults)
    preprocessor.feature_names_in_  # ensures this attribute exists after fit
    ohe = preprocessor.named_transformers_["cat"]
    cat_categories = {}
    for col_name, cats in zip(categorical_features, ohe.categories_ if categorical_features else []):
        cat_categories[col_name] = [str(c) for c in cats]

    metadata = {
        "feature_order": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_defaults": {col: float(X_train[col].mean()) for col in numeric_features},
        "categorical_defaults": {col: str(X_train[col].mode(dropna=True).iloc[0]) if not X_train[col].mode().empty else "" for col in categorical_features},
        "categorical_categories": cat_categories
    }

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    pickle.dump(model, open("artifacts/model.pkl", "wb"))
    pickle.dump(preprocessor, open("artifacts/preprocessor.pkl", "wb"))
    pickle.dump((X_test, y_test), open("artifacts/test_data.pkl", "wb"))
    pickle.dump(metadata, open("artifacts/metadata.pkl", "wb"))

    print("\n💾 Saved files in ./artifacts:")
    print(" - model.pkl")
    print(" - preprocessor.pkl")
    print(" - test_data.pkl")
    print(" - metadata.pkl")

if __name__ == "__main__":
    main()
