from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import re
import os

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# Configurations - Ensure these match train_model.py if used for inference
MODEL_DIR = "spam_model"

# Load the model and vectorizer
try:
    vectorizer = tf.saved_model.load(os.path.join(MODEL_DIR, 'vectorizer'))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model.keras'))
    print("Model and vectorizer loaded successfully for API.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    print("Please ensure train_model.py has been run and saved the model correctly.")
    exit(1)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,$]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = []
    important_short_words = {'$', '£', '€', 'win', 'won', 'txt', 'msg', 'ur', 'u', 'r'}
    for word in words:
        if len(word) >= 2 or word.lower() in important_short_words:
            filtered_words.append(word)
    return ' '.join(filtered_words)


def predict_message(message):
    print(f"[DEBUG] Incoming message: {message}")
    # Rule 1: If message contains "http://" (case-insensitive), it's spam.
    if re.search(r'http://', message, re.IGNORECASE):
        print("[DEBUG] HTTP link found! Returning SPAM.")
        return True

    preprocessed = preprocess_text(message)
    print(f"[DEBUG] Preprocessed message: {preprocessed}")
    vectorized = vectorizer([preprocessed]).numpy()
    pred = model.predict(vectorized, verbose=0)[0][0]
    print(f"[DEBUG] Initial model prediction (pred): {pred}")

    # Rule 2: Increase probability for spam-related words
    spam_keywords = ['free', 'win', 'cash', 'prize', 'urgent', 'claim', 'deal', 'sex', 'xxx', 'viagra', 'loan', 'credit', 'money back', 'guarantee', 'congratulations', 'lottery', 'offer', 'exclusive', 'limited time']
    contains_spam_keywords = False
    for keyword in spam_keywords:
        if keyword in preprocessed:
            contains_spam_keywords = True
            break

    if contains_spam_keywords and pred < 0.7: # If spam keywords are present and prediction is not already high spam
        pred += 0.2 # Increase probability by 0.2
        print(f"[DEBUG] Pred after keyword adjustment: {pred}")

    final_prediction = pred > 0.5
    print(f"[DEBUG] Final prediction result: {final_prediction} (pred > 0.5)")
    return final_prediction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['text'] # Changed from data['message'] to data['text']
    is_spam = predict_message(message)
    # The 'pred' variable is local to predict_message, so we need to pass it or re-calculate it for confidence if needed
    # For now, let's just return is_spam
    return jsonify({'is_spam': is_spam})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)