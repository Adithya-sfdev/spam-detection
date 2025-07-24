import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './Auth.css';
import { auth, googleProvider, signInWithEmailAndPassword, signInWithPopup } from '../firebase';

const Login = ({ onLoginSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const navigate = useNavigate();

  const handleGoogleSuccess = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      const userData = {
        name: user.displayName,
        email: user.email,
        picture: user.photoURL,
        isAuthenticated: true,
        isVerified: user.emailVerified,
      };
      localStorage.setItem('user', JSON.stringify(userData));
      onLoginSuccess(userData);
      navigate('/dashboard');
    } catch (error) {
      alert('Google login failed: ' + error.message);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;
      const userData = {
        email: user.email,
        name: user.displayName || email.split('@')[0],
        isAuthenticated: true,
        isVerified: user.emailVerified,
      };
      localStorage.setItem('user', JSON.stringify(userData));
      onLoginSuccess(userData);
      navigate('/dashboard');
    } catch (error) {
      alert('Login failed: ' + error.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h1>Welcome Back</h1>
          <p>Sign in to your account</p>
        </div>
        <div className="google-auth-section">
          <button onClick={handleGoogleSuccess} type="button" className="gsi-material-button">
            <div className="gsi-material-button-content-wrapper">
              <img
                className="gsi-material-button-icon"
                src="https://developers.google.com/identity/images/g-logo.png"
                alt="Google logo"
              />
              <span className="gsi-material-button-contents">Sign in with Google</span>
            </div>
          </button>
        </div>
        <div className="divider">
          <span>or</span>
        </div>
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type={isPasswordVisible ? 'text' : 'password'}
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <span
                className="password-toggle"
                onClick={() => setIsPasswordVisible(!isPasswordVisible)}
              >
                {isPasswordVisible ? '👁️' : '🙈'}
              </span>
            </div>
          </div>
          <button type="submit" className="auth-button">Sign In</button>
        </form>
        <div className="auth-footer">
          <p>Don't have an account? <Link to="/register">Sign up</Link></p>
        </div>
      </div>
    </div>
  );
};

export default Login;
