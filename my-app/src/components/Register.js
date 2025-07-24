import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './Auth.css';
import { auth, googleProvider, createUserWithEmailAndPassword, signInWithPopup } from '../firebase';
import { updateProfile, sendEmailVerification } from 'firebase/auth';

const Register = ({ onLoginSuccess }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [isRegistered, setIsRegistered] = useState(false);
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isConfirmPasswordVisible, setIsConfirmPasswordVisible] = useState(false);
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
      alert('Google registration failed: ' + error.message);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      alert('Passwords do not match!');
      return;
    }
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;
      await updateProfile(user, { displayName: formData.name });
      await sendEmailVerification(user);
      const userData = {
        name: formData.name,
        email: user.email,
        isAuthenticated: true,
        isVerified: false,
      };
      localStorage.setItem('user', JSON.stringify(userData));
      setIsRegistered(true);
      alert('Registration successful! Please check your email for a verification link.');
    } catch (error) {
      alert('Registration failed: ' + error.message);
    }
  };

  if (isRegistered) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h1>Check your email!</h1>
          <p>A verification email has been sent to <strong>{formData.email}</strong>.</p>
          <p>Please verify your email address by clicking the link in the email. You may now sign in.</p>
          <div className="auth-footer">
            <p>Didn't receive the email? <button className="auth-link">Resend</button></p>
            <p><Link to="/login">Back to sign in</Link></p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h1>Create Account</h1>
          <p>Join us today</p>
        </div>
        <div className="google-auth-section">
          <button
            onClick={handleGoogleSuccess}
            type="button"
            className="gsi-material-button"
          >
            <div className="gsi-material-button-content-wrapper">
              <img
                className="gsi-material-button-icon"
                src="https://developers.google.com/identity/images/g-logo.png"
                alt="Google logo"
              />
              <span className="gsi-material-button-contents">Register with Google</span>
            </div>
          </button>
        </div>
        <div className="divider"><span>or</span></div>
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type="text"
                name="name"
                placeholder="Full Name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
          </div>
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type="email"
                name="email"
                placeholder="Email address"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
          </div>
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type={isPasswordVisible ? 'text' : 'password'}
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
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
          <div className="input-group">
            <div className="input-wrapper">
              <input
                type={isConfirmPasswordVisible ? 'text' : 'password'}
                name="confirmPassword"
                placeholder="Confirm Password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
              />
              <span
                className="password-toggle"
                onClick={() => setIsConfirmPasswordVisible(!isConfirmPasswordVisible)}
              >
                {isConfirmPasswordVisible ? '👁️' : '🙈'}
              </span>
            </div>
          </div>
          <button type="submit" className="auth-button">Create Account</button>
        </form>
        <div className="auth-footer">
          <p>Already have an account? <Link to="/login">Sign in</Link></p>
        </div>
      </div>
    </div>
  );
};

export default Register;
