# =========================
# app.py
# =========================

# ---- standard imports ----
import numpy as np
import joblib
from flask import Flask, render_template, request

# ---- create flask app ----
app = Flask(__name__)

print("APP FILE STARTED")

# ---- load saved ML objects ----
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

print("Model, scaler, and features loaded")

# -------------------------
# HOME PAGE
# -------------------------
@app.route("/")
def home():
    """
    Shows the input form page
    """
    return render_template("index.html", features=features)


# -------------------------
# PREDICTION ROUTE
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Takes input from form, runs model prediction,
    and shows result on inner-page.html
    """

    # 1. Get values from form
    input_values = []

    for feature in features:
        value = request.form.get(feature)
        input_values.append(float(value))

    # 2. Convert to numpy array
    input_array = np.array(input_values).reshape(1, -1)

    # 3. Scale input
    scaled_input = scaler.transform(input_array)

    # 4. Predict
    prediction = model.predict(scaled_input)[0]

    # 5. Convert prediction to readable label
    if prediction == 1:
        result = "STARFORMING"
    else:
        result = "STARBURST"

    # 6. Send result to output page
    return render_template(
        "inner-page.html",
        prediction=result
    )


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    print("INSIDE MAIN BLOCK")
    app.run(debug=True, port=2222)
