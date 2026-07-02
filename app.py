from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import csv
import os

app = Flask(__name__)

# ================= MODEL LOAD =================
model = joblib.load("fraud_model.pkl")

HISTORY_FILE = "history.csv"

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ================= PREDICTION =================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result = "FRAUD DETECTED" if prediction else "TRANSACTION SAFE"

        # ================= REASON LOGIC =================
        reasons = []

        if float(data.get("amount", 0)) > 5000:
            reasons.append("High Amount")

        if int(data.get("attempt_count", 0)) > 3:
            reasons.append("Multiple Attempts")

        if data.get("is_night_transaction"):
            reasons.append("Night Transaction")

        reason_text = ", ".join(reasons) if reasons else "Normal"

        # ================= SAVE HISTORY =================
        file_exists = os.path.isfile(HISTORY_FILE)

        with open(HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["Amount", "Attempts", "Result", "Probability", "Reason"])

            writer.writerow([
                data.get("amount"),
                data.get("attempt_count"),
                result,
                round(prob * 100, 2),
                reason_text
            ])

        return jsonify({
            "is_suspicious": bool(prediction),
            "result": result,
            "fraud_probability": round(prob * 100, 2),
            "reason": reason_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ================= HISTORY API =================

@app.route("/history-data")
def history_data():
    data = []

    try:
        with open(HISTORY_FILE, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except:
        pass

    return jsonify(data)


# ================= RUN =================

if __name__ == "__main__":
    print("🚀 NEW CLEAN BACKEND RUNNING")
    app.run(debug=True)
