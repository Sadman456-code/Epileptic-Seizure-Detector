from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form["eeg_data"]
        values = np.array([float(x) for x in input_data.strip().split(",")])

        if len(values) != 178:
            return "Error: Exactly 178 EEG values are required."

        # MOCK prediction logic (replace this later with real model)
        avg = np.mean(values)
        if avg > 0.5:  # arbitrary threshold just for testing
            label = "Epileptic"
            confidence = 87.5
        else:
            label = "Non-Epileptic"
            confidence = 94.2

        return f"Mock Prediction: {label} (Confidence: {confidence}%)"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
