import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';

const Dashboard = ({ user, onLogout }) => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [predictionDetails, setPredictionDetails] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking'); // Initial state is 'checking'
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Redirect if not authenticated
  useEffect(() => {
    if (!user?.isAuthenticated) {
      navigate('/login');
    }
    checkApiHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, navigate]);

  // Check backend API status
  const checkApiHealth = async () => {
    try {
      const response = await fetch('http://localhost:5000/', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const data = await response.json();
        setApiStatus(data.model_loaded ? 'model_loaded' : 'fallback_only');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
    }
  };

  // Call backend or use fallback detection
  const detectSpam = async (text) => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = await response.json();
        const isSpam = data.is_spam;
        const confidence = data.confidence !== undefined ? `${(data.confidence * 100).toFixed(1)}%` : 'N/A';

        return {
          prediction: isSpam ? 'Spam' : 'Not Spam',
          confidence: confidence,
          details: { ...data, confidence },
        };
      }
      throw new Error('API request failed');
    } catch (error) {
      return enhancedFallbackDetection(text);
    }
  };

  // Client-side spam detection (keywords)
  const enhancedFallbackDetection = (text) => {
    const spamKeywords = [
      'free', 'win', 'money', 'cash', 'prize', 'urgent', 'act now',
      'limited time', 'call now', 'click here', 'congratulations'
    ];
    const lowerText = text.toLowerCase();
    const keywordMatches = spamKeywords.filter(keyword =>
      lowerText.includes(keyword)
    ).length;
    const isSpam = keywordMatches >= 2 ||
      lowerText.includes('give me money') ||
      /win.*\$\d+/.test(lowerText);
    const confidence = Math.min(0.85 + (keywordMatches * 0.05), 0.95);
    return {
      prediction: isSpam ? 'Spam' : 'Not Spam',
      confidence: `${(confidence * 100).toFixed(1)}%`,
      details: { method: 'client_fallback', keyword_matches: keywordMatches, confidence: `${(confidence * 100).toFixed(1)}%` },
    };
  };

  // Handle spam check submission
  const handleCheck = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to check');
      return;
    }
    setIsLoading(true);
    setError('');
    setResult('');
    setPredictionDetails(null);
    try {
      const detectionResult = await detectSpam(inputText);
      setResult(detectionResult.prediction);
      setPredictionDetails(detectionResult.details);
    } catch (err) {
      setError('Failed to analyze the message');
    } finally {
      setIsLoading(false);
    }
  };

  // Logout handler
  const handleLogout = () => {
    localStorage.removeItem('user');
    onLogout();
    navigate('/login');
  };

  // API status display logic
  const getApiStatusDisplay = () => {
    switch (apiStatus) {
      case 'model_loaded':
        return { text: 'ML Model Active', color: '#2ed573', icon: '🤖' };
      case 'fallback_only':
        return { text: 'Fallback Mode', color: '#ffa502', icon: '⚡' };
      default:
        // This case now correctly handles 'offline' and any other unexpected states
        return { text: 'Offline Mode', color: '#ff4757', icon: '📱' };
    }
  };

  const statusDisplay = getApiStatusDisplay();

  return (
    <div className="dashboard-container">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="navbar-left">
          <img src="/logo192.png" alt="Logo" className="navbar-logo" />
          <h2>Advanced Spam Detector</h2>
        </div>
        <div className="navbar-right">
          {/* 
            THE FIX IS HERE:
            Only show the status indicator if the API status is NOT offline AND NOT checking.
          */}
          {apiStatus !== 'offline' && apiStatus !== 'checking' && (
            <div className="api-status" style={{ backgroundColor: statusDisplay.color }}>
              {statusDisplay.icon} {statusDisplay.text}
            </div>
          )}
          <button onClick={handleLogout} className="logout-button">Logout</button>
        </div>
      </nav>

      {/* Main content area for centering the card */}
      <div className="main-content">
        <div className="detection-card">
          <h1>Spam Detection System</h1>

          <div className="input-section">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter a message to check for spam...&#10;&#10;Example: 'Congratulations you win a cash prize!' or 'Hello, how are you?'"
              rows="5"
            />
            <button
              onClick={handleCheck}
              className="check-button"
              disabled={isLoading || !inputText.trim()}
            >
              {isLoading ? '🔄 Analyzing...' : '🔍 Check for Spam'}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="error-message">❌ {error}</div>
          )}

          {/* Result */}
          {result && (
            <div className={`result ${result.toLowerCase().replace(' ', '-')}`}>
              <div className="result-header">
                <span className="result-icon">{result === 'Spam' ? '🚨' : '✅'}</span>
                <span className="result-text">{result}</span>
                {predictionDetails?.confidence && (
                  <span className="confidence">({predictionDetails.confidence} confidence)</span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
