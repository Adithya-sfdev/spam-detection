import pickle
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import your class and preprocessing from train_model.py
from train_model import AdvancedSpamDetector, SENTENCE_TRANSFORMERS_AVAILABLE

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Load config, tokenizer, model
with open('advanced_model_config.pickle', 'rb') as f:
    config = pickle.load(f)
with open('advanced_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
model = keras.models.load_model('advanced_spam_model.h5')

# Detector for preprocessing
detector = AdvancedSpamDetector()

def predict_spam(input_text):
    # Preprocess
    processed = detector.advanced_preprocess_text(input_text)
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=config['max_len'], padding='post')
    # Embedding (real or dummy)
    if os.path.exists('sentence_transformer_model') and SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
        sentence_transformer = SentenceTransformer('sentence_transformer_model')
        emb = sentence_transformer.encode([processed], convert_to_numpy=True)
    else:
        emb = np.random.rand(1, config['embedding_dim'])
    # Try hybrid or fallback
    try:
        pred = model.predict([pad, emb])[0][0]
    except:
        pred = model.predict(pad)[0][0]
    label = "spam" if pred > 0.5 else "ham"
    print(f"\nPrediction: {label} (score: {pred:.3f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        predict_spam(text)
    else:
        text = input("Enter text to check if spam or ham: ")
        predict_spam(text)
