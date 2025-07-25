# Spam Detection System
## _AI-Powered, Modern, and User-Friendly_

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-blue?logo=react)](https://react.dev/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Spam Detection Demo](https://github.com/Adithya-sfdev/spam-detection/blob/main/my-app/public/logo512.png)

The Spam Detection System is a full-stack, machine learning-powered web app to detect spam messages in real time.  
It features a beautiful React frontend, secure Firebase authentication, and a robust Flask + TensorFlow backend.

- Paste or type any message
- Instantly see if it’s spam or not
- ✨ Powered by AI ✨

---

## Features

- Real-time spam detection (ML model: LSTM, TensorFlow)
- REST API backend (Flask)
- User authentication (Email/Password & Google via Firebase)
- Responsive, modern UI (React)
- Client-side fallback detection if backend is offline
- Keyword-based spam heuristics for extra accuracy
- Easy local setup & deployment

---

## Tech Stack

- [React](https://react.dev/) - Frontend
- [Firebase Auth](https://firebase.google.com/) - Authentication
- [Flask](https://flask.palletsprojects.com/) - Backend API
- [TensorFlow](https://www.tensorflow.org/) - ML Model
- [Scikit-learn](https://scikit-learn.org/) - Data processing
- [Python](https://www.python.org/) - Backend language

---

## Installation

### Backend (Flask + ML)
```sh
cd backend
pip install -r requirements.txt
python train_model.py      # Train & save the model (only needed once)
python api.py              # Start the API (http://localhost:5000)
```

### Frontend (React)
```sh
cd my-app
npm install
npm start                  # Runs on http://localhost:3000
```

---

## Usage

1. Register/Login (Email/Password or Google)
2. Go to Dashboard
3. Paste or type a message to check for spam
4. Get instant results (Spam/Not Spam, with confidence)

---

## Screenshots

![Dashboard Screenshot](https://user-images.githubusercontent.com/yourusername/dashboard.png)
![Spam Result Example](https://user-images.githubusercontent.com/yourusername/spam-result.png)

---

## Model Details

- Preprocessing: Lowercasing, removing emails/phones, special chars, etc.
- Model: Bidirectional LSTM, trained on SMS spam dataset
- Extra rules: Keyword boosting, HTTP link detection

---

## Contributing

Want to contribute? Great!

1. Fork this repo
2. Create a new branch (`feature/your-feature`)
3. Commit your changes
4. Open a Pull Request

---

## License

MIT

**Free Software, Hell Yeah!**

---

## Acknowledgements

- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [TensorFlow](https://www.tensorflow.org/)
- [Create React App](https://create-react-app.dev/)
- [Firebase](https://firebase.google.com/)

---

> Made with ❤️ for learning and real-world spam protection!
