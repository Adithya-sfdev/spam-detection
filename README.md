# Enhanced AI Spam Detection System

## 🚀 Deployment Instructions

### Frontend (Vercel) Deployment

1. **Environment Variables Setup in Vercel:**
   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add the following variables:

```
REACT_APP_FIREBASE_API_KEY=your_firebase_api_key
REACT_APP_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your_project_id
REACT_APP_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
REACT_APP_FIREBASE_APP_ID=your_app_id
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id
REACT_APP_BACKEND_URL=https://your-backend-url.vercel.app
```

2. **Deploy Frontend:**
   ```bash
   cd my-app
   npm install
   npm run build
   # Deploy to Vercel
   ```

### Backend (Vercel) Deployment

1. **Deploy Backend:**
   ```bash
   cd backend
   # Deploy to Vercel using Vercel CLI or GitHub integration
   ```

2. **Backend URL:**
   - After deployment, get your backend URL from Vercel
   - Update the `REACT_APP_BACKEND_URL` in frontend environment variables

## 🔧 Local Development

### Frontend
```bash
cd my-app
npm install
npm start
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python api.py
```

## 📁 Project Structure

```
spam detection/
├── backend/
│   ├── api.py                 # Main Flask API
│   ├── requirements.txt       # Python dependencies
│   ├── vercel.json           # Backend Vercel config
│   └── sentence_transformer_model/  # AI model files
├── my-app/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── App.js           # Main app component
│   │   └── firebase.js      # Firebase configuration
│   ├── package.json         # Node.js dependencies
│   └── vercel.json          # Frontend Vercel config
└── README.md
```

## 🛠️ Troubleshooting

### Common Issues:

1. **"Could not connect to Enhanced AI analysis server"**
   - Check if backend URL is correct in environment variables
   - Ensure backend is deployed and running
   - Verify CORS configuration

2. **Firebase Authentication Issues**
   - Verify Firebase environment variables are set correctly
   - Check Firebase project configuration
   - Ensure Google OAuth client ID is correct

3. **CORS Errors**
   - Backend CORS is configured for production domains
   - Check if your domain is in the allowed origins list

## 🔐 Security Notes

- Never commit `.env` files to version control
- Use environment variables for all sensitive configuration
- Enable Firebase Authentication rules
- Configure CORS properly for production

## 📞 Support

For issues related to:
- Frontend: Check Vercel deployment logs
- Backend: Check Vercel function logs
- AI Model: Ensure model files are in the correct location
