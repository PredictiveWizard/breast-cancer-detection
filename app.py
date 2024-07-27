import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# List of feature names
FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(request.form[feature]) for feature in FEATURES]
    prediction = model.predict([input_features])
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    return render_template('index.html', features=FEATURES, result=result)

if __name__ == '__main__':
    app.run(debug=True)
