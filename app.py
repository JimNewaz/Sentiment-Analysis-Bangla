from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and preprocessing objects
model = joblib.load('best_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
feature_selector = joblib.load('feature_selector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Transform text using the saved TF-IDF vectorizer
    text_transformed = tfidf_vectorizer.transform([text])

    # Select important features
    text_transformed = text_transformed[:, feature_selector]

    # Predict using the loaded model
    prediction = model.predict(text_transformed)[0]
    probability = model.predict_proba(text_transformed)[0][1]

    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
