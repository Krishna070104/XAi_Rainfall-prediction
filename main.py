# main.py

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate
from src.explain import explain_model
from src.predict import predict_with_explanation


def main():
    print("🚀 Starting Rainfall XAI Project\n")

    # =========================
    # 1. LOAD DATA
    # =========================
    print("📥 Loading dataset...")
    data = load_data("data/heavy_rain_dataset.csv")

    if data is None:
        print("❌ Failed to load data. Exiting...")
        return

    # =========================
    # 2. PREPROCESS DATA
    # =========================
    print("\n🧹 Preprocessing data...")
    X, y = preprocess_data(data)

    # =========================
    # 3. TRAIN MODEL
    # =========================
    print("\n🤖 Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # =========================
    # 4. EVALUATE MODEL
    # =========================
    print("\n📊 Evaluating model...")
    evaluate(model, X_test, y_test)

    # =========================
    # 5. EXPLAIN MODEL (SHAP)
    # =========================
    print("\n🔍 Generating SHAP explanations...")
    explain_model(model, X_train, X_test)

    # =========================
    # 6. PREDICT + TEXT EXPLANATION
    # =========================
    print("\n🧠 Generating prediction with explanation...\n")
    predict_with_explanation()

    print("\n✅ Project Completed Successfully!")


# =========================
# RUN PROJECT
# =========================
if __name__ == "__main__":
    main()