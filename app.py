from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    weight = float(data['Weight'])
    length1 = float(data['Length1'])
    length2 = float(data['Length2'])
    length3 = float(data['Length3'])
    height = float(data['Height'])
    width = float(data['Width'])

    # Create the feature array
    features = np.array([[weight, length1, length2, length3, height, width]])

    # Make the prediction
    species_encoded = model.predict(features)[0]
    species = le.inverse_transform([species_encoded])[0]

    return render_template('index.html', prediction_text=f'Predicted Species: {species}')

if __name__ == '__main__':
    app.run(debug=True)
