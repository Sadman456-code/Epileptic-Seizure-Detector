from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", message=None, status=None, eeg_data="")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.get("eeg_data", "").strip()
    try:
        if not input_data:
            return render_template(
                "index.html",
                message="⚠️ Please enter the EEG values to get the prediction.",
                status="error",
                eeg_data=input_data
            )

        values = np.array([float(x) for x in input_data.split(",")])

        if len(values) != 178:
            return render_template(
                "index.html",
                message="❌ Error: Exactly 178 EEG values are required.",
                status="error",
                eeg_data=input_data
            )

        # MOCK prediction logic (replace this with actual model)
        avg = np.mean(values)
        if avg > 0.5:
            label = "🧠 Epileptic"
            confidence = 87.5
        else:
            label = "✅ Non-Epileptic"
            confidence = 94.2

        result_message = f"Prediction: <strong>{label}</strong><br>Confidence: <strong>{confidence:.1f}%</strong>"
        return render_template(
            "index.html",
            message=result_message,
            status="success",
            eeg_data=input_data
        )

    except ValueError:
        return render_template(
            "index.html",
            message="❌ Error: Please enter only comma-separated numbers.",
            status="error",
            eeg_data=input_data
        )
    except Exception as e:
        return render_template(
            "index.html",
            message=f"❌ Unexpected Error: {str(e)}",
            status="error",
            eeg_data=input_data
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
