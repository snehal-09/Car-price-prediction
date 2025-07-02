from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = int(request.form['fuel_type'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])

        features = np.array([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]])
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Estimated Car Price: ₹{round(prediction[0], 2)} Lakh")

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
