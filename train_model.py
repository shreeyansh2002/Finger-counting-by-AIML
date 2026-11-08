"""
train_model.py
Trains a supervised classifier on the hand-gesture dataset collected
with collect_data.py, evaluates it with a held-out test split, and
saves the trained model for finger_counter_ai.py to load.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier5
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "hand_data.csv"
MODEL_FILE = "hand_gesture_model.pkl"


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples across classes: {sorted(df['label'].unique())}")

    X = df.drop(columns=["label"]).values
    y = df["label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc * 100:.1f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_names = df.drop(columns=["label"]).columns
    importances = model.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda t: -t[1]):
        print(f"  {name:20s} {imp:.3f}")

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
