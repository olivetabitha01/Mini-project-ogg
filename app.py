from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/alz_model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        inputs = []
        for i in range(1, 8):
            inputs.append([
                float(request.form[f"alpha{i}"]),
                float(request.form[f"theta{i}"]),
                float(request.form[f"step{i}"]),
                float(request.form[f"blink{i}"]),
                float(request.form[f"hr{i}"]),
                float(request.form[f"temp{i}"]),
                float(request.form[f"abeta{i}"])
            ])
        df = pd.DataFrame(inputs, columns=[
            "alpha_wave_ratio", "theta_wave_ratio", "step_variance",
            "blink_rate", "heart_rate", "temperature", "abeta_level"
        ])
        predictions = model.predict(df)
        risk_score = round(predictions.mean(), 2)
        final_result = "LOW" if risk_score < 0.5 else "MODERATE" if risk_score < 1.5 else "HIGH"
        return render_template("index.html", result=final_result, preds=predictions.tolist())
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
