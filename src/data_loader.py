import pandas as pd

def load_data(path):
    try:
        data = pd.read_csv(path)
        print("✅ Data loaded successfully")
        return data
    except Exception as e:
        print("❌ Error loading data:", e)
        return None
