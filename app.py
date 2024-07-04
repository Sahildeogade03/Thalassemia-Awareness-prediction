from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", 'rb')) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = request.form
        features = [
            float(data['RBC']),
            float(data['HGB']),
            float(data['MCV']),
            float(data['MCH']),
            float(data['MCHC'])
        ]
        features = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features)
        
        return jsonify({'prediction': (prediction[0])})
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)