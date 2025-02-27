import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_waterfall(gbc, x):
    explainer = shap.Explainer(gbc)
    shap_values = explainer(x)

    # Create a waterfall plot for the first sample
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10)  # Show top 10 features

    # Save the plot
    waterfall_path = "static/shap_waterfall.png"
    plt.savefig(waterfall_path)
    plt.close()

    return waterfall_path  # Return the path to be used in Flask

print(dir())  # Print all available functions
