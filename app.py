from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load(r'C:\Users\ASUS\OneDrive\Documents\PI\model\web\model.pkl')

# Mapping manual sesuai label encoding waktu training
label_map = {
    0: 'low risk',
    1: 'mid risk',
    2: 'high risk'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age        = float(request.form['age'])
        systolic   = float(request.form['systolic'])
        diastolic  = float(request.form['diastolic'])
        bs_mgdl    = float(request.form['bs'])
        bs         = bs_mgdl / 18   # mg/dL -> mmol/L
        body_temp_c = float(request.form['body_temp'])
        body_temp  = (body_temp_c * 9/5) + 32  # fahrenheit to celcius
        heart_rate = float(request.form['heart_rate'])

        input_data = pd.DataFrame(
            [[age, systolic, diastolic, bs, body_temp, heart_rate]],
            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        )

        prediction    = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        result    = label_map[int(prediction)]
        prob_low  = round(probabilities[0] * 100, 1)
        prob_mid  = round(probabilities[1] * 100, 1)
        prob_high = round(probabilities[2] * 100, 1)

        return render_template('result.html',
                               result=result,
                               prob_low=prob_low,
                               prob_mid=prob_mid,
                               prob_high=prob_high)
    except Exception as e:
        return (f"Terjadi kesalahan: {e}", 400)

if __name__ == '__main__':
    app.run(debug=True)
