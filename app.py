from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", message=None, status=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form["eeg_data"]
        values = np.array([float(x) for x in input_data.strip().split(",")])

        if len(values) != 178:
            return render_template("index.html", message="❌ Error: Exactly 178 EEG values are required.", status="error")

        # MOCK prediction logic (replace this with actual model)
        avg = np.mean(values)
        if avg > 0.5:
            label = "🧠 Epileptic"
            confidence = 87.5
        else:
            label = "✅ Non-Epileptic"
            confidence = 94.2

        result_message = f"Prediction: <strong>{label}</strong><br>Confidence: <strong>{confidence:.1f}%</strong>"
        return render_template("index.html", message=result_message, status="success")

    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}", status="error")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
