from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss'
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/xgb_model.pkl")

    print("✅ Model trained and saved")

    return model, X_train, X_test, y_train, y_test