import pandas as pd

def preprocess_data(data):

    # Rename columns (based on your dataset)
    data.columns = [
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "cloud_cover",
        "rainfall_24h",
        "soil_moisture",
        "target"
    ]

    # Check missing values
    print("\nMissing values:\n", data.isnull().sum())

    # Fill missing values if any
    data = data.fillna(data.mean(numeric_only=True))

    # Feature / target split
    X = data.drop("target", axis=1)
    y = data["target"]

    print("✅ Preprocessing completed")

    return X, y