import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("data.csv")

# 2. Drop non-predictive columns
df = df.drop(columns=["index", "Patient Id"])

# 3. Define features and target
target = "Level"
X = df.drop(columns=[target])
y = df[target].str.strip().str.capitalize()  # normalize labels

# 4. Explicitly define feature types
num_feats = [col for col in X.columns if col != "Gender"]
cat_feats = ["Gender"]

# 5. Preprocessing
numeric_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_feats),
        ("cat", categorical_tf, cat_feats)
    ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Balance training data with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 8. Model
rf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42
    ))
])

# 9. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring="f1_macro")
print("CV F1_macro:", scores.mean(), "+/-", scores.std())

# 10. Train and evaluate
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Save model
joblib.dump(rf, "model_rf.joblib")
print("✅ Model saved as model_rf.joblib")