from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model and training data for SHAP
model = joblib.load("models/xgb_model.pkl")

# Load a sample of training data for SHAP background
try:
    import pandas as pd
    from src.data_loader import load_data
    from src.preprocess import preprocess_data
    data = load_data("data/heavy_rain_dataset.csv")
    X, y = preprocess_data(data)
    X_train = X
except Exception as e:
    print("Warning: Could not load training data for SHAP:", e)
    X_train = None

FEATURES = [
    ("temperature", "Temperature", 0, 50, "°C"),
    ("humidity", "Humidity", 0, 100, "%"),
    ("wind_speed", "Wind Speed", 0, 100, "km/h"),
    ("pressure", "Pressure", 900, 1100, "hPa"),
    ("cloud_cover", "Cloud Cover", 0, 100, "%"),
    ("rainfall_24h", "Rainfall 24h", 0, 500, "mm"),
    ("soil_moisture", "Soil Moisture", 0, 100, "%")
]


def get_risk_level(score):
    if score > 1:
        return "HIGH", "danger"
    elif score > 0:
        return "MODERATE", "warning"
    else:
        return "LOW", "success"


def generate_shap_explanation(model, X_train, sample_df):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(sample_df)

    values = shap_values.values[0]
    features = sample_df.columns

    explanation = []
    for i in range(len(values)):
        feature = features[i]
        value = sample_df.iloc[0, i]
        impact = values[i]
        if impact > 0:
            explanation.append({
                "text": f"{feature} ({value}) increased rainfall risk",
                "impact": "positive",
                "value": float(impact)
            })
        else:
            explanation.append({
                "text": f"{feature} ({value}) reduced rainfall risk",
                "impact": "negative",
                "value": float(impact)
            })

    total_score = float(sum(values))
    return explanation, total_score


def generate_shap_waterfall(model, X_train, sample_df, filename):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(sample_df)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"static/{filename}")
    plt.close()


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_dict = {}
        for key, _, min_val, max_val, _ in FEATURES:
            value = float(request.form.get(key, 0))
            if not (min_val <= value <= max_val):
                return jsonify({
                    "error": f"{key} must be between {min_val} and {max_val}"
                }), 400
            user_dict[key] = value

        sample_df = pd.DataFrame([user_dict])

        prediction = model.predict(sample_df)[0]
        prediction_proba = model.predict_proba(sample_df)[0].tolist()

        explanation, score = generate_shap_explanation(model, X_train, sample_df)
        risk_level, risk_class = get_risk_level(score)

        # Generate waterfall plot for this prediction
        plot_filename = f"shap_waterfall_{hash(str(user_dict.values()))}.png"
        generate_shap_waterfall(model, X_train, sample_df, plot_filename)

        result = {
            "prediction": int(prediction),
            "prediction_label": "Heavy Rainfall Expected" if prediction == 1 else "No Heavy Rainfall",
            "probability": prediction_proba,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "score": round(score, 4),
            "explanation": explanation,
            "waterfall_plot": plot_filename
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shap_summary")
def shap_summary():
    return render_template("shap_summary.html")


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, port=5000)

