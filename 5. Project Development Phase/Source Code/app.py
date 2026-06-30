from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("floods.save")


# Home Page
@app.route('/')
def home():
    return render_template('home.html')


# Prediction Input Page
@app.route('/predict')
def predict():
    return render_template('predict.html')


# Prediction Result
@app.route('/result', methods=['POST'])
def result():
    try:
        values = [
            float(request.form['Temp']),
            float(request.form['Humidity']),
            float(request.form['cloud_cover']),
            float(request.form['ANNUAL']),
            float(request.form['jan_feb']),
            float(request.form['mar_may']),
            float(request.form['jun_sep']),
            float(request.form['oct_dec']),
            float(request.form['avgjune']),
            float(request.form['sub'])
        ]

        prediction = model.predict([values])[0]

        if prediction == 1:
            return render_template('flood.html')
        else:
            return render_template('noflood.html')

    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)