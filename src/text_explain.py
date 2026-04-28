import shap
import numpy as np

def generate_text_explanation(model, X_train, X_sample):

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample)

    values = shap_values.values[0]
    features = X_sample.columns

    explanation = []

    for i in range(len(values)):
        feature = features[i]
        value = X_sample.iloc[0, i]
        impact = values[i]

        if impact > 0:
            explanation.append(f"{feature} ({value}) increased rainfall risk")
        else:
            explanation.append(f"{feature} ({value}) reduced rainfall risk")

    return explanation, sum(values)