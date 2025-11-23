import random
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import io
import base64
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend for rendering plots
import matplotlib.pyplot as plt

# ---- Global Font Settings (Normal) ----
plt.rcParams.update({
    'font.size': 14,        # Axis values
    'axes.titlesize': 18,   # Plot titles
    'axes.labelsize': 16,   # X and Y labels
    'xtick.labelsize': 14,  # X tick labels
    'ytick.labelsize': 14,  # Y tick labels
    'legend.fontsize': 14   # Legend text
})

app = Flask(__name__)
cnn = tf.keras.models.load_model("1dcnn_model.keras")
sc = joblib.load("scaler.pkl")  # Load the scaler

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("eeg_file")
        if not file:
            return render_template("index.html", message="⚠️ Please upload EEG data.", status="error")
        
        filename = file.filename

        raw_data = file.read().decode("utf-8")
        data_str = raw_data.replace("\n", " ").replace(",", " ")
        values = np.array([float(x) for x in data_str.split() if x.strip() != ""])

        if len(values) != 4097:
            return render_template("index.html", message="❌ File must contain exactly 4097 EEG values.", status="error")

        # Segment and scale
        segment_length = 178
        max_start_multiple = ((len(values) - segment_length) // segment_length) * segment_length
        valid_starts = list(range(0, max_start_multiple + 1, segment_length))
        start = random.choice(valid_starts)

        segment = values[start:start + segment_length]
        segment_scaled = sc.transform(segment.reshape(1, -1)).reshape(1, segment_length, 1)

        # Prediction
        y_pred = cnn.predict(segment_scaled)
        is_epileptic = y_pred[0][0] > 0.5
        label = "🧠 Epileptic" if is_epileptic else "✅ Non-Epileptic"
        confidence = float(y_pred[0][0]) * 100 if is_epileptic else (1 - float(y_pred[0][0])) * 100
        result_message = f"Prediction: <strong>{label}</strong><br>Confidence: <strong>{confidence:.1f}%</strong>"

        # Load reference data
        df = pd.read_csv('Epileptic Seizure Recognition.csv')
        df1 = df.drop(columns=['Unnamed', 'y'])
        epileptic_curve = df1.iloc[6663, :].values
        non_epileptic_curve = df1.iloc[6619, :].values
        time = np.linspace(0, 1, 178)

        # Calculate global y-axis limits
        global_min = min(segment.min(), epileptic_curve.min(), non_epileptic_curve.min())
        global_max = max(segment.max(), epileptic_curve.max(), non_epileptic_curve.max())

        # Plot: User EEG Segment
        fig_user, ax_user = plt.subplots(figsize=(12, 10))
        ax_user.plot(time, segment, color='blue', linewidth=2)
        ax_user.set_title("User EEG")
        ax_user.set_xlabel("Time (Seconds)")
        ax_user.set_ylabel("Amplitude (Voltage)")
        ax_user.set_ylim(global_min, global_max)
        ax_user.tick_params(axis='both', which='major')
        for tick in ax_user.get_xticklabels() + ax_user.get_yticklabels():
            tick.set_fontweight('bold')

        plt.tight_layout()
        user_img_io = io.BytesIO()
        plt.savefig(user_img_io, format='png')
        user_img_io.seek(0)
        user_plot_base64 = base64.b64encode(user_img_io.read()).decode('utf-8')
        plt.close(fig_user)

        # Plot: Reference EEG Patterns
        fig_ref, ax_ref = plt.subplots(figsize=(12, 10))
        ax_ref.plot(time, epileptic_curve, label='Epileptic', color='red', linewidth=2)
        ax_ref.plot(time, non_epileptic_curve, label='Non-Epileptic', color='green', linewidth=2)
        ax_ref.set_title("Reference EEG")
        ax_ref.set_xlabel("Time (Seconds)")
        ax_ref.set_ylabel("Amplitude (Voltage)")
        ax_ref.set_ylim(global_min, global_max)
        legend = ax_ref.legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')

        ax_ref.tick_params(axis='both', which='major')
        for tick in ax_ref.get_xticklabels() + ax_ref.get_yticklabels():
            tick.set_fontweight('bold')

        plt.tight_layout()
        ref_img_io = io.BytesIO()
        plt.savefig(ref_img_io, format='png')
        ref_img_io.seek(0)
        ref_plot_base64 = base64.b64encode(ref_img_io.read()).decode('utf-8')
        plt.close(fig_ref)

        return render_template("index.html",
                               message=result_message,
                               status="success",
                               user_plot=user_plot_base64,
                               ref_plot=ref_plot_base64,
                               filename=filename)

    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", message="❌ Please upload the required EEG data file or check the data format.", status="error")

if __name__ == "__main__":
    app.run(debug=False)
