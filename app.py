from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/alz_model.joblib")

# store latest sensor data (auto from ESP32)
sensor_data = {
    "blink_rate": 0,
    "heart_rate": 0,
    "temperature": 0,
    "step_variance": 0
}

# store manual user inputs
manual_data = {
    "alpha_wave_ratio": 0.7,
    "theta_wave_ratio": 0.5,
    "abeta_level": 1.0
}

latest_prediction = {"risk": None, "score": None}


@app.route("/")
def index():
    return render_template(
        "index.html",
        sensor=sensor_data,
        manual=manual_data,
        prediction=latest_prediction
    )


# ESP32 sends sensor data here every 10s
@app.route("/update", methods=["POST"])
def update_sensor_data():
    global sensor_data, latest_prediction
    data = request.get_json(force=True)

    for key in data:
        if key in sensor_data:
            sensor_data[key] = data[key]

    # Combine manual + sensor data for prediction
    combined = {**manual_data, **sensor_data}
    df = pd.DataFrame([combined])
    score = float(model.predict(df)[0])
    risk = "LOW" if score < 0.5 else "MODERATE" if score < 1.5 else "HIGH"
    latest_prediction = {"risk": risk, "score": round(score, 2)}

    return jsonify({"status": "updated", "prediction": latest_prediction})


# User manually updates alpha/theta/abeta
@app.route("/manual", methods=["POST"])
def manual_update():
    global manual_data, latest_prediction
    data = request.form
    try:
        manual_data["alpha_wave_ratio"] = float(data["alpha_wave_ratio"])
        manual_data["theta_wave_ratio"] = float(data["theta_wave_ratio"])
        manual_data["abeta_level"] = float(data["abeta_level"])
    except:
        pass

    # Combine for prediction
    combined = {**manual_data, **sensor_data}
    df = pd.DataFrame([combined])
    score = float(model.predict(df)[0])
    risk = "LOW" if score < 0.5 else "MODERATE" if score < 1.5 else "HIGH"
    latest_prediction = {"risk": risk, "score": round(score, 2)}

    return render_template(
        "index.html",
        sensor=sensor_data,
        manual=manual_data,
        prediction=latest_prediction
    )


@app.route("/get_data")
def get_data():
    return jsonify({"sensor": sensor_data, "prediction": latest_prediction})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
