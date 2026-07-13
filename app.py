from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])

    data = np.array([[hours, attendance]])
    print(data)
    predicted_score = model.predict(data)[0]

    if predicted_score >= 60:
        output = "PASS 🎉"
    else:
        output = "FAIL ❌"

    return render_template(
        "index.html",
        prediction=round(predicted_score, 2),
        result=output
    )

if __name__ == "__main__":
    app.run(debug=True)
    

