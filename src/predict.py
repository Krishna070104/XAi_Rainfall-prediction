import joblib
import pandas as pd
from src.text_explain import generate_text_explanation

def predict_with_explanation():

    model = joblib.load("models/xgb_model.pkl")

    print("\n🌤️ Enter current weather conditions:")
    print("   (Press Enter after each value)\n")

    # Expected features in model order
    expected_features = ["temperature", "humidity", "wind_speed", "pressure", "cloud_cover", "rainfall_24h", "soil_moisture"]

    user_dict = {}
    for feature in expected_features:
        while True:
            try:
                value = float(input(f"   {feature.replace('_', ' ').title()}: "))
                # Basic validation
                if feature == "temperature":
                    if not 0 <= value <= 50: raise ValueError
                elif feature == "humidity":
                    if not 0 <= value <= 100: raise ValueError
                elif feature == "wind_speed":
                    if not 0 <= value <= 100: raise ValueError
                elif feature == "pressure":
                    if not 900 <= value <= 1100: raise ValueError
                elif feature == "cloud_cover":
                    if not 0 <= value <= 100: raise ValueError
                elif feature == "rainfall_24h":
                    if not 0 <= value <= 500: raise ValueError
                elif feature == "soil_moisture":
                    if not 0 <= value <= 100: raise ValueError
                user_dict[feature] = value
                break
            except ValueError:
                examples = {
                    "temperature": "30.5",
                    "humidity": "65.0",
                    "wind_speed": "15.0",
                    "pressure": "1013.0",
                    "cloud_cover": "40.0",
                    "rainfall_24h": "12.5",
                    "soil_moisture": "30.0"
                }
                print(f"   ⚠️ Invalid value for {feature}. Try again (e.g., {feature}: {examples[feature]})")

    sample_df = pd.DataFrame([user_dict])

    # Verify feature order matches model
    assert list(sample_df.columns) == expected_features, "Feature order mismatch with model!"

    prediction = model.predict(sample_df)[0]

    # Generate explanation
    explanation, score = generate_text_explanation(model, sample_df, sample_df)

    print("\n🔍 Prediction Result:\n")

    if prediction == 1:
        print("🌧️ Heavy Rainfall Expected\n")
    else:
        print("☀️ No Heavy Rainfall\n")

    print("📊 Explanation:\n")

    for exp in explanation:
        print("•", exp)

    # Risk level
    if score > 1:
        print("\n⚠️ Risk Level: HIGH")
    elif score > 0:
        print("\n⚠️ Risk Level: MODERATE")
    else:
        print("\n⚠️ Risk Level: LOW")