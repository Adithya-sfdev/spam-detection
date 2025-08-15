import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import { REACT_APP_API_BASE } from '../config';

const Dashboard = ({ user, onLogout }) => {
    const [inputText, setInputText] = useState('');
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [showDetails, setShowDetails] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        if (!user?.isAuthenticated) {
            navigate('/login');
        }
    }, [user, navigate]);

    const handleCheck = async () => {
        if (!inputText.trim()) {
            setError('Please enter some text to check.');
            return;
        }
        
        setIsLoading(true);
        setError('');
        setResult(null);
        setShowDetails(false);

        console.log('Sending request with text:', inputText);

        try {
            const response = await fetch(`${REACT_APP_API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText }),
            });

            console.log('Response status:', response.status);
            
            const data = await response.json();
            console.log('Enhanced API Response:', data);

            if (!response.ok) {
                throw new Error(data.error || 'An API error occurred.');
            }
            
            if (!data.prediction) {
                console.error('Invalid API response - missing prediction field:', data);
                setError('Invalid response from server - missing prediction data.');
                return;
            }

            setResult(data);
            console.log('Enhanced result set to:', data);

        } catch (err) {
            console.error("Fetch Error:", err);
            setError(`Error: Could not connect to the Enhanced AI analysis server. Please ensure the backend is running. [${err.message}]`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('user');
        onLogout();
        navigate('/login');
    };

    // Get intent emoji based on analysis
    const getIntentEmoji = (intent) => {
        const intentEmojis = {
            'greeting': '👋',
            'business': '💼',
            'personal': '👤',
            'gratitude': '🙏',
            'neutral': '💬'
        };
        return intentEmojis[intent] || '💬';
    };

    return (
        <div className="dashboard-container">
            <nav className="navbar">
                <div className="navbar-left">
                    <img src={`${process.env.PUBLIC_URL}/logo192.png`} alt="Logo" className="navbar-logo" />
                    <h2>Advanced AI Spam Detection</h2>
                </div>
                <div className="navbar-right">
                    <button onClick={handleLogout} className="logout-button">Logout</button>
                </div>
            </nav>

            <div className="main-content">
                <div className="detection-card">
                    <h1>🤖 Advanced AI Spam Detection</h1>
                    
                    <div className="input-section">
                        <textarea
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            placeholder="Enter a message to analyze for spam using advanced AI contextual understanding..."
                            rows="5"
                        />
                        <button
                            onClick={handleCheck}
                            className="check-button enhanced"
                            disabled={isLoading || !inputText.trim()}
                        >
                            {isLoading ? '🧠 AI Analyzing...' : '🔍 Analyze with AI'}
                        </button>

                        {/* Prototype Note */}
                        <p style={{
                            fontSize: '0.85rem',
                            color: '#888',
                            marginTop: '6px',
                            fontStyle: 'italic'
                        }}>
                            Note*: This is a prototype model — results may vary.
                        </p>
                    </div>

                    {error && (<div className="error-message">❌ {error}</div>)}

                    {result && !error && (
                        <div className={`result enhanced ${result.prediction.toLowerCase().replace(' ', '-')}`}>
                            <div className="result-header">
                                <span className="result-icon">
                                    {result.prediction === 'Spam' ? '🚨' : '✅'}
                                </span>
                                <span className="result-text">{result.prediction}</span>
                            </div>
                            
                            {/* Enhanced AI Explanation */}
                            {result.explanation && (
                                <div className="ai-explanation">
                                    <div className="explanation-icon">🧠</div>
                                    <div className="explanation-text">{result.explanation}</div>
                                </div>
                            )}

                            {/* Analysis Summary */}
                            {result.analysis && (
                                <div className="analysis-summary">
                                    <div className="analysis-item">
                                        <span className="analysis-label">
                                            {getIntentEmoji(result.analysis.intent)} Intent:
                                        </span>
                                        <span className="analysis-value">{result.analysis.intent || 'neutral'}</span>
                                    </div>
                                </div>
                            )}

                            {/* Toggle for detailed analysis */}
                            {(result.analysis?.confidence_factors?.length > 0 || result.model_info) && (
                                <div className="details-toggle">
                                    <button 
                                        onClick={() => setShowDetails(!showDetails)}
                                        className="toggle-details-btn"
                                    >
                                        {showDetails ? '▼ Hide Details' : '▶ Show AI Analysis Details'}
                                    </button>
                                </div>
                            )}

                            {/* Detailed Analysis (collapsible) */}
                            {showDetails && (
                                <div className="detailed-analysis">
                                    {result.analysis?.confidence_factors?.length > 0 && (
                                        <div className="confidence-factors">
                                            <h4>🔍 AI Analysis Factors:</h4>
                                            <ul>
                                                {result.analysis.confidence_factors.map((factor, index) => (
                                                    <li key={index}>{factor}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}

                                    {result.analysis?.context?.length > 0 && (
                                        <div className="context-info">
                                            <h4>📝 Context Analysis:</h4>
                                            <div className="context-tags">
                                                {result.analysis.context.map((ctx, index) => (
                                                    <span key={index} className="context-tag">{ctx}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {result.model_info && (
                                        <div className="model-info">
                                            <h4>🤖 AI Model Information:</h4>
                                            <div className="model-details">
                                                <span>Architecture: {result.model_info.architecture}</span>
                                                <span>Training Accuracy: {result.model_info.training_accuracy}</span>
                                                {result.model_info.semantic_analysis && (
                                                    <span>✅ Semantic Analysis: Enabled</span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>     
            </div>
        </div>
    );
};

export default Dashboard;
