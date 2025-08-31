import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv",encoding='latin1') # <-- change filename to your dataset

# Clean dataset
df.rename(columns={"ï»¿Age": "Age"}, inplace=True)
df.drop(["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"], axis=1, inplace=True)

# Define target column
target = "Attrition"   # <-- change if needed
X = df.drop(target, axis=1)
y = df[target]

# Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Pipeline = preprocessing + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save trained pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names too
with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Model and features saved!")
