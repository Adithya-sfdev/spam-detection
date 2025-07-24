import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import os

# Configurations
MODEL_DIR = "spam_model"
DATA_FILE = "spam.csv"
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128

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

# Load and preprocess data
try:
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    print(f"Dataset loaded successfully with {len(df)} rows.")
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    exit(1)

df = df[['v1', 'v2']].copy()
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)
df['processed_message'] = df['message'].apply(preprocess_text)
df = df[df['processed_message'].str.len() > 0].copy()

messages = df['processed_message'].values
labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)

# TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None
)
vectorizer.adapt(X_train)

# Model that accepts tokenized input
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=MAX_FEATURES+1, output_dim=EMBEDDING_DIM, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_vec = vectorizer(X_train).numpy()
X_test_vec = vectorizer(X_test).numpy()

history = model.fit(
    X_train_vec, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test_vec, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
    ],
    verbose=1
)

# Save the vectorizer and model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
tf.saved_model.save(vectorizer, os.path.join(MODEL_DIR, 'vectorizer'))
model.save(os.path.join(MODEL_DIR, 'model.keras'))
print("\nModel and vectorizer saved successfully!")

def predict_message(message):
    # Rule 1: If message contains "http://" (case-insensitive), it's spam.
    if re.search(r'http://', message, re.IGNORECASE):
        return True

    preprocessed = preprocess_text(message)
    vectorized = vectorizer([preprocessed]).numpy()
    pred = model.predict(vectorized, verbose=0)[0][0]

    # Rule 2: Increase probability for spam-related words
    spam_keywords = ['free', 'win', 'cash', 'prize', 'urgent', 'claim', 'deal', 'sex', 'xxx', 'viagra', 'loan', 'credit', 'money back', 'guarantee', 'congratulations', 'lottery', 'offer', 'exclusive', 'limited time']
    contains_spam_keywords = False
    for keyword in spam_keywords:
        if keyword in preprocessed:
            contains_spam_keywords = True
            break

    if contains_spam_keywords and pred < 0.7: # If spam keywords are present and prediction is not already high spam
        pred += 0.2 # Increase probability by 0.2

    return pred > 0.5

# Interactive prediction loop
print("\nModel ready for prediction! Type your message (or 'quit' to exit):")
while True:
    user_input = input("Message: ")
    if user_input.lower() == 'quit':
        break
    is_spam = predict_message(user_input)
    print(f"  Prediction: {'SPAM' if is_spam else 'HAM'}")
