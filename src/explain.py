import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X_train, X_test):

    print("🔍 Generating SHAP explanations...")

    # Create outputs folder
    os.makedirs("output", exist_ok=True)

    # SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("output/shap_summary.png")
    plt.close()

    print("✅ SHAP summary saved")

    # Optional: Feature importance bar
    shap.plots.bar(shap_values, show=False)
    plt.savefig("output/shap_bar.png")
    plt.close()