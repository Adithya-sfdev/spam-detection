# Advanced Spam Detection System
## _AI-Powered, Modern, and User-Friendly_

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-blue?logo=react)](https://react.dev/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modern, full-stack Spam Detection System using Machine Learning (TensorFlow), Flask API, and a beautiful React frontend with Firebase Authentication.

---

## 🚀 Features
- **Real-time Spam Detection** using a trained ML model (LSTM, TensorFlow)
- **REST API** backend with Flask
- **User Authentication** (Email/Password & Google via Firebase)
- **Attractive, Responsive UI** (React)
- **Client-side fallback detection** if backend is offline
- **Keyword-based spam heuristics** for extra accuracy
- **Easy to deploy** (Docker-ready, local setup)

---

## 🛠️ Tech Stack
- **Frontend:** React, Firebase Auth, CSS
- **Backend:** Python, Flask, TensorFlow, Scikit-learn
- **ML Model:** LSTM Neural Network (Keras)
- **Data:** Enron Spam Dataset

---

## 📦 Project Structure
```
spam-detection/
  backend/         # Flask API + ML model
    api.py
    train_model.py
    advanced_spam_model.h5    # Saved model
    advanced_tokenizer.pickle # Tokenizer
    enron_spam_data.csv      # Training data
  my-app/          # React frontend
    src/
      components/  # Login, Register, Dashboard
      firebase.js  # Firebase config
  README.md        # (This file)
```

---

## ⚡ Quick Start

### 1. Backend (Flask + ML)
```bash
cd backend
pip install -r requirements.txt
python train_model.py            # Train & save the model (only needed once)
python api.py                    # Start the API (default: http://localhost:5000)
```

### 2. Frontend (React)
```bash
cd my-app
npm install
npm start                        # Runs on http://localhost:3000
```

---

## 🧑‍💻 Usage
1. **Register/Login** (Email/Password or Google)
2. **Go to Dashboard**
3. **Paste or type a message** to check for spam
4. **Get instant results** (Spam/Not Spam, with confidence)

---

## 📊 Model Details
- Preprocessing: Lowercasing, removing emails/phones, special chars, etc.
- Model: Bidirectional LSTM, trained on Enron spam dataset
- Extra rules: Keyword boosting, HTTP link detection
- Advanced features: Contextual analysis, intent detection

---

## 🤝 Contributing
1. Fork this repo
2. Create a new branch (`feature/your-feature`)
3. Commit your changes
4. Open a Pull Request

---

## 📄 License
MIT

---

## 🙏 Acknowledgements
- [Enron Spam Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- [TensorFlow](https://www.tensorflow.org/)
- [Create React App](https://create-react-app.dev/)
- [Firebase](https://firebase.google.com/)

---

> Made with ❤️ for learning and real-world spam protection!
