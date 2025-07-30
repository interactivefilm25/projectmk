import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# === Load CSV directly ===
CSV_PATH = "data/combined/labeled_features_with_emotions.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# === Validate structure ===
if "emotion" not in df.columns:
    raise ValueError("Missing 'emotion' label column in the dataset")

# === Prepare features and labels ===
X = df.drop(columns=["emotion"])
y = df["emotion"]

# Convert any non-numeric feature values to numbers (coerce errors to NaN)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Train ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Save artifacts ===
joblib.dump(model, "emotion_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\n✅ Model and label encoder saved as 'emotion_model.pkl' and 'label_encoder.pkl'")