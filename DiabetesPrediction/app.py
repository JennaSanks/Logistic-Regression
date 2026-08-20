from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Logistic Regression model
model = pickle.load(open("DiabetesModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read two inputs
    glucose = float(request.form['glucose'])
    bmi = float(request.form['bmi'])

    # Create input array
    input_data = np.array([[
        glucose,
        bmi
    ]])

    # Prediction
    prediction = model.predict(input_data)

    # Binary output
    if prediction[0] == 1:
        result = "Prediction: Diabetes"
    else:
        result = "Prediction: No Diabetes"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)