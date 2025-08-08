# Advanced AI Spam Detection (Full‑Stack)

AI‑powered spam detection with a Python/Flask backend and a modern React frontend. Lightweight deployment (TFLite runtime) for fast boots on Render, and static hosting on GitHub Pages or Vercel for the UI.

Live (GitHub Pages): [`adithya-sfdev.github.io/spam-detection`](https://adithya-sfdev.github.io/spam-detection/)

---

## Features
- Real‑time spam prediction via REST API (`/predict`)
- Enhanced semantic/heuristic analysis with detailed explanations
- React UI with Firebase auth (email/password + Google)
- CORS‑enabled API for cross‑origin frontends
- Health endpoints: `/`, `/health`, `/debug`

---

## Tech Stack
- Frontend: React 18, React Router, Firebase Auth, CSS
- Backend: Python 3.11, Flask, Gunicorn, NumPy, TFLite Runtime
- Model: LSTM hybrid architecture (Keras); runtime served via `.tflite`

---

## Project Structure
```
spam detection/
  api/                       # Flask API service
    api.py                   # App entry (exposes /predict, /health, ...)

  backend/                   # Training / experimentation (local only)
    train_model.py
    predict_spam.py
    enron_spam_data.csv

  my-app/                    # React frontend
    src/                     # Components, styles
    package.json             # gh‑pages + build scripts

  advanced_spam_model.tflite # Runtime model picked by API
  advanced_tokenizer.pickle  # Tokenizer
  advanced_model_config.pickle

  Procfile                   # For Render (Gunicorn)
  requirements.txt           # Slim deps (uses tflite-runtime)
  runtime.txt                # Python version for Render
```

---

## Run locally

### Backend (Flask)
```bash
pip install -r requirements.txt
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:5000 api.api:app
# or for quick dev: python api/api.py
```
Model loading order: `advanced_spam_model.tflite` ➜ `spam_model.tflite` ➜ rebuilt `.h5` weights (if TensorFlow present).

### Frontend (React)
```bash
cd my-app
npm install
echo REACT_APP_API_BASE=http://localhost:5000 > .env.local
npm start
```

---

## Deploy

### Backend on Render
- Build: `pip install -r requirements.txt`
- Start (Procfile): `web: gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT api.api:app`
- Place `advanced_spam_model.tflite`, `advanced_tokenizer.pickle`, `advanced_model_config.pickle` in repo root.
- Health check path: `/health`

### Frontend on GitHub Pages
`my-app/package.json` already contains `predeploy`/`deploy` scripts.
```bash
cd my-app
echo REACT_APP_API_BASE=https://<your-render-service>.onrender.com > .env.production
npm run deploy
```
Site: `https://adithya-sfdev.github.io/spam-detection/`

### Frontend on Vercel (optional)
- Project Root: `my-app/`
- Build Command: `npm run build`
- Output Directory: `build`
- Env var: `REACT_APP_API_BASE=https://<your-render-service>.onrender.com`

---

## API
- POST `/predict` → `{ text: string }` → `{ prediction, explanation, analysis, model_info }`
- GET `/health` → service status
- GET `/` and `/debug` → metadata and diagnostics

---

## Acknowledgements
- Enron Spam Dataset
- TensorFlow / Keras
- Create React App / React Router
- Firebase Auth

---

MIT License
