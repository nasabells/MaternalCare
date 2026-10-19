from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load(r'C:\Users\ASUS\OneDrive\Documents\PI\model\web\model.pkl')

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

def validasi_input(data):
    errors = []
    if not (10 <= data['age'] <= 70):
        errors.append("Usia ibu harus antara 10-70 tahun")
    if not (36.7 <= data['body_temp_c'] <= 39.4):
        errors.append("Suhu tubuh harus antara 36.7-39.4°C")
    if not (70 <= data['systolic'] <= 160):
        errors.append("Tekanan sistolik harus antara 70-160 mmHg")
    if not (49 <= data['diastolic'] <= 100):
        errors.append("Tekanan diastolik harus antara 49-100 mmHg")
    if not (108 <= data['bs_mgdl'] <= 342):
        errors.append("Gula darah harus antara 108-342 mg/dL")
    if not (60 <= data['heart_rate'] <= 90):
        errors.append("Detak jantung harus antara 60-90 bpm")
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age         = float(request.form['age'])
        systolic    = float(request.form['systolic'])
        diastolic   = float(request.form['diastolic'])
        bs_mgdl     = float(request.form['bs'])
        body_temp_c = float(request.form['body_temp'])
        heart_rate  = float(request.form['heart_rate'])

        # kumpulkan data mentah (belum dikonversi) untuk divalidasi dulu
        raw_data = {
            'age': age,
            'systolic': systolic,
            'diastolic': diastolic,
            'bs_mgdl': bs_mgdl,
            'body_temp_c': body_temp_c,
            'heart_rate': heart_rate
        }

        errors = validasi_input(raw_data)
        if errors:
            return render_template('index.html', errors=errors)

        # kalau lolos validasi, baru konversi satuan
        bs = bs_mgdl / 18                     # mg/dL -> mmol/L
        body_temp = (body_temp_c * 9/5) + 32  # Celsius -> Fahrenheit

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
    app.run(debug=True, use_reloader=False)