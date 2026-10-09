from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import csv
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = [
    "hour",
    "day_of_week",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3"
]

ranges = {
    "hour": (0, 23),
    "day_of_week": (0, 6),
    "voltage": (200, 260),
    "global_intensity": (0, 50),
    "sub_metering_1": (0, 100),
    "sub_metering_2": (0, 100),
    "sub_metering_3": (0, 100)
}

def valid_features(data):
    values = []

    for f in features:
        if f not in data or not isinstance(data[f], (int, float)):
            return None

        min_value, max_value = ranges[f]
        if data[f] < min_value or data[f] > max_value:
            return None

        values.append(data[f])

    return values

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = valid_features(data)

    if values is None:
        return jsonify({"error": "Please enter values in the allowed ranges"}), 400

    prediction = model.predict([values])[0]

    return jsonify({
        "prediction": float(prediction),
        "message": "Predicted energy consumption: " + str(round(prediction, 2)) + " kWh"
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    input_features = data.get("features", {})
    actual = data.get("actual")

    if not isinstance(actual, (int, float)):
        return jsonify({"error": "Actual value must be a number"}), 400

    row = valid_features(input_features)

    if row is None:
        return jsonify({"error": "Feature values must be in the allowed ranges"}), 400

    file_exists = os.path.exists("feedback.csv")

    with open("feedback.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(features + ["actual"])
        writer.writerow(row + [actual])

    return jsonify({"message": "Feedback saved"})

if __name__ == "__main__":
    app.run(debug=True)
