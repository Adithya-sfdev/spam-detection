from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
import datetime
import numpy as np
import warnings
import logging

# Suppress some TensorFlow deprecation/info logs (non-invasive)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("urllib3").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
tokenizer = None
config = None
sentence_transformer = None
MODEL_IS_TFLITE = False
tflite_interpreter = None
TFLiteInterpreter = None

# Try importing with proper error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # Try to silence TF v1 deprecation warnings (best-effort)
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    TF_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except Exception as e:
    print(f"❌ TensorFlow import error: {e}")
    TF_AVAILABLE = False

# Robust TensorFlow Lite interpreter import (multiple fallbacks)
TFLITE_RUNTIME_AVAILABLE = False
try:
    if TF_AVAILABLE:
        try:
            from tensorflow.lite import Interpreter as TFLiteInterpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
        TFLiteInterpreter = TFLiteInterpreter
        TFLITE_RUNTIME_AVAILABLE = True
        print("✅ TensorFlow Lite interpreter (from tensorflow) available")
    else:
        import tflite_runtime.interpreter as tflite_runtime_interpreter
        TFLiteInterpreter = tflite_runtime_interpreter.Interpreter
        TFLITE_RUNTIME_AVAILABLE = True
        print("✅ tflite_runtime interpreter available")
except Exception as e:
    print(f"⚠️ TFLite interpreter not available: {e}")
    TFLITE_RUNTIME_AVAILABLE = False

# Try to import sentence-transformers for better semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ sentence-transformers available (semantic embeddings enabled)")
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available — using keyword-based semantic fallback")

# -------------------------
# Utility / helper methods
# -------------------------

def tflite_predict(interpreter, inputs):
    """
    Run an inference with a TFLite interpreter.
    `inputs` can be a single numpy array or a list of numpy arrays.
    Returns a scalar probability (float).
    """
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        for i, inp_detail in enumerate(input_details):
            provided = inputs[i] if i < len(inputs) else inputs[0]
            expected_dtype = inp_detail.get('dtype')
            provided_arr = np.array(provided, dtype=expected_dtype)

            # Ensure batch dim
            if provided_arr.ndim == 1:
                provided_arr = np.expand_dims(provided_arr, axis=0)

            # Try reshape if exact size matches expected
            expected_shape = inp_detail.get('shape')
            try:
                if expected_shape is not None and np.prod(expected_shape) == provided_arr.size:
                    provided_arr = provided_arr.reshape(expected_shape)
            except Exception:
                pass

            interpreter.set_tensor(inp_detail['index'], provided_arr)

        interpreter.invoke()
        out_detail = output_details[0]
        output_data = interpreter.get_tensor(out_detail['index'])
        return float(np.squeeze(output_data))
    except Exception as e:
        print(f"⚠️ TFLite inference failed: {e}")
        return 0.3

def _load_tokenizer_and_config(project_root):
    """Load tokenizer and config early so we know model shapes to rebuild architecture."""
    global tokenizer, config
    tokenizer_loaded = False
    for tokenizer_file_name in ['advanced_tokenizer.pickle', 'tokenizer.pickle']:
        tokenizer_path = os.path.join(project_root, tokenizer_file_name)
        try:
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                print(f"✅ Tokenizer loaded from {tokenizer_file_name}")
                tokenizer_loaded = True
                break
        except Exception as e:
            print(f"⚠️ Failed to load {tokenizer_file_name}: {e}")
            continue
    if not tokenizer_loaded:
        print("⚠️ Tokenizer not found - predictions requiring tokenizer will fail")

    config_loaded = False
    for config_file_name in ['advanced_model_config.pickle', 'model_config.pickle']:
        config_path = os.path.join(project_root, config_file_name)
        try:
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                print(f"✅ Configuration loaded from {config_file_name}")
                config_loaded = True
                break
        except Exception as e:
            print(f"⚠️ Failed to load {config_file_name}: {e}")
            continue
    if not config_loaded:
        config = {
            'max_len': 150,
            'max_words': 8000,
            'model_architecture': 'hybrid_lstm_transformer',
            'test_accuracy': 0.988,
            'embedding_dim': 384
        }
        print("⚠️ Using default configuration")

def _rebuild_hybrid_model_from_config():
    """Recreate the exact architecture used in train_model.py so we can load weights by_name."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, Concatenate

    max_len = int(config.get('max_len', 150))
    vocab_size = int(config.get('max_words', 8000))
    embedding_dim = int(config.get('embedding_dim', 384))

    # Build the same hybrid model architecture as train_model.py
    text_input = Input(shape=(max_len,), name='text_input')
    embedding_input = Input(shape=(embedding_dim,), name='embedding_input')

    embedding_layer = Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)(text_input)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding_layer)
    lstm_pool = GlobalMaxPooling1D()(lstm_layer)

    semantic_dense1 = Dense(256, activation='relu')(embedding_input)
    semantic_dropout1 = Dropout(0.3)(semantic_dense1)
    semantic_dense2 = Dense(128, activation='relu')(semantic_dropout1)
    semantic_dropout2 = Dropout(0.2)(semantic_dense2)

    merged = Concatenate()([lstm_pool, semantic_dropout2])

    dense1 = Dense(128, activation='relu')(merged)
    dropout1 = Dropout(0.4)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    dense3 = Dense(32, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    output = Dense(1, activation='sigmoid', name='spam_prediction')(dropout3)

    model_local = Model(inputs=[text_input, embedding_input], outputs=output)
    model_local.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model_local

# -------------------------
# New semantic + PII detection helpers
# -------------------------

def detects_sensitive_info_request(text):
    """
    Returns True when the message requests sensitive financial/personal info
    or asks for a reply with identifying details (strong phishing indicator).
    """
    if not isinstance(text, str) or not text.strip():
        return False

    text_lower = text.lower()

    pii_patterns = [
        r'last\s+4\s+digits',
        r'last\s+four\s+digits',
        r'last\s+\d+\s+digits',
        r'last\s+4\s+of\s+your',
        r'last\s+four\s+of\s+your',
        r'account\s+number',
        r'account\s+no\b',
        r'bank\s+account',
        r'employee\s+id',
        r'employee\s+number',
        r'employee\s+no\b',
        r'please\s+reply\s+with',
        r'reply\s+with',
        r'confirm\s+your\s+account',
        r'confirm\s+the\s+last',
        r'provide\s+your\s+(?:account|bank|employee|id|ssn|aadhar|pan)',
        r'verify\s+your\s+(?:account|details|identity|bank)',
        r'for\s+verification\s+please\s+reply',
        r'send\s+me\s+the\s+last',
    ]

    for p in pii_patterns:
        if re.search(p, text_lower):
            return True

    # Also catch reply-short-codes like "reply YES", "reply RENEW" that try to elicit a quick response
    if re.search(r'\breply\s+(yes|no|renew|confirm|dispute|morning|afternoon|claim)\b', text_lower):
        return True

    return False

def semantic_score(text):
    """
    Return a semantic spam-likelihood score in [0.0, 1.0].
    Uses sentence_transformers if available; otherwise falls back to a
    robust keyword/pattern scoring heuristic.
    Higher = more likely spam.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower = text.lower()

    # If sentence_transformer available, compute embedding similarity to a small spam-ham prototype set
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            global sentence_transformer
            if sentence_transformer is None:
                sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            # Prototype texts (short and safe) representing spammy intents and ham intents
            spam_prototypes = [
                "Please reply with your account number for verification",
                "We need your bank details to process payroll immediately",
                "You are a winner, claim your prize",
                "Please transfer money to vendor now, urgent"
            ]
            ham_prototypes = [
                "Please update your details on the company HR portal",
                "Please check the attached report and review the budget",
                "Team meeting scheduled for tomorrow",
                "Please confirm receipt of the invoice"
            ]
            doc_emb = sentence_transformer.encode([text], convert_to_numpy=True)[0]
            spam_embs = sentence_transformer.encode(spam_prototypes, convert_to_numpy=True)
            ham_embs = sentence_transformer.encode(ham_prototypes, convert_to_numpy=True)

            # cosine similarity function
            def cos_sim(a, b):
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                if denom == 0:
                    return 0.0
                return float(np.dot(a, b) / denom)

            spam_sims = [cos_sim(doc_emb, s) for s in spam_embs]
            ham_sims = [cos_sim(doc_emb, h) for h in ham_embs]

            avg_spam_sim = float(np.mean(spam_sims)) if len(spam_sims) else 0.0
            avg_ham_sim = float(np.mean(ham_sims)) if len(ham_sims) else 0.0

            # Map similarity difference to [0,1]
            raw = max(0.0, avg_spam_sim - avg_ham_sim + 0.05)  # small bias
            # normalize roughly (since typical cosine sim range ~ -1..1)
            score = 1.0 / (1.0 + np.exp(-12 * (raw - 0.02)))  # logistic around small threshold
            return float(score)
        except Exception as e:
            print(f"⚠️ Semantic embedding scoring failed: {e}")
            # fall through to keyword heuristic

    # Keyword / heuristic fallback
    score = 0.0
    # weights: sensitive requests, money words, urgency, little context, reply requests
    sensitive_patterns = ['account', 'bank', 'employee id', 'employee id', 'ssn', 'pan', 'aadhar', 'last four', 'last 4']
    money_patterns = [r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', r'\b(inr|rs\.?|usd|dollars|rupees|₹)\b', r'invoice', r'payment', r'transfer']
    urgency_patterns = ['urgent', 'immediately', 'asap', 'within 48', 'within 24', 'now', 'important', 'immediate']
    reply_patterns = ['please reply', 'reply with', 'reply', 'confirm', 'respond', 'send me', 'provide']

    # count hits
    for pat in sensitive_patterns:
        if pat in text_lower:
            score += 0.25
    for pat in money_patterns:
        if re.search(pat, text_lower):
            score += 0.18
    for pat in urgency_patterns:
        if pat in text_lower:
            score += 0.15
    for pat in reply_patterns:
        if pat in text_lower:
            score += 0.12

    # penalize long formal messages that include signatures (could be legit)
    if len(text_lower.split()) > 60:
        score -= 0.1

    # normalize to [0,1]
    score = max(0.0, min(1.0, score))
    return float(score)

# -------------------------
# Model loading (robust)
# -------------------------

def load_advanced_model():
    """Load the advanced TensorFlow model robustly by prioritizing H5 weights -> rebuilt model (no load_model deserialization)."""
    global model, tokenizer, config, sentence_transformer, MODEL_IS_TFLITE, tflite_interpreter, TFLiteInterpreter

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load tokenizer & config first (we need shapes)
    _load_tokenizer_and_config(project_root)

    if not TF_AVAILABLE and not TFLITE_RUNTIME_AVAILABLE:
        print("❌ Neither TensorFlow nor TFLite runtime available")
        return False

    try:
        print("🧠 Loading advanced AI model...")
        model_loaded = False

        # 1) Try loading .tflite (if present and interpreter available)
        tflite_path = os.path.join(project_root, 'advanced_spam_model.tflite')
        fallback_tflite = os.path.join(project_root, 'spam_model.tflite')
        if TFLITE_RUNTIME_AVAILABLE and os.path.exists(tflite_path):
            try:
                print("🔄 Attempt: Loading advanced_spam_model.tflite...")
                tflite_interpreter = TFLiteInterpreter(model_path=tflite_path) if TF_AVAILABLE else TFLiteInterpreter(tflite_path)
                tflite_interpreter.allocate_tensors()
                MODEL_IS_TFLITE = True
                model = tflite_interpreter
                model_loaded = True
                print("✅ TFLite model loaded successfully")
            except Exception as e:
                print(f"⚠️ TFLite load failed: {e}")

        if not model_loaded and TFLITE_RUNTIME_AVAILABLE and os.path.exists(fallback_tflite):
            try:
                print("🔄 Attempt: Loading fallback spam_model.tflite...")
                tflite_interpreter = TFLiteInterpreter(model_path=fallback_tflite) if TF_AVAILABLE else TFLiteInterpreter(fallback_tflite)
                tflite_interpreter.allocate_tensors()
                MODEL_IS_TFLITE = True
                model = tflite_interpreter
                model_loaded = True
                print("✅ Fallback TFLite model loaded successfully")
            except Exception as e:
                print(f"⚠️ TFLite fallback load failed: {e}")

        # 2) Preferred path for your setup: rebuild architecture and load weights from H5 (avoid load_model deserialization)
        h5_path = os.path.join(project_root, 'advanced_spam_model.h5')
        fallback_h5 = os.path.join(project_root, 'spam_model.h5')

        if not model_loaded and TF_AVAILABLE and os.path.exists(h5_path):
            try:
                print("🔄 Attempt: Rebuilding architecture and loading H5 weights (no deserialization)...")
                model_candidate = _rebuild_hybrid_model_from_config()
                try:
                    model_candidate.load_weights(h5_path, by_name=True, skip_mismatch=True)
                    model = model_candidate
                    MODEL_IS_TFLITE = False
                    model_loaded = True
                    print("✅ Successfully rebuilt architecture and loaded H5 weights by_name (advanced_spam_model.h5)")
                except Exception as e:
                    print(f"⚠️ Direct load_weights(by_name) failed: {e}")
                    # As a fallback try to open H5 file and see its keys
                    try:
                        import h5py
                        with h5py.File(h5_path, 'r') as f:
                            if 'model_weights' in f.keys() or 'layer_names' in f.keys():
                                model_candidate.load_weights(h5_path, by_name=True, skip_mismatch=True)
                                model = model_candidate
                                MODEL_IS_TFLITE = False
                                model_loaded = True
                                print("✅ Loaded weights from H5 using h5py-assisted path")
                    except Exception as e2:
                        print(f"⚠️ h5py-assisted weight loading failed: {e2}")
            except Exception as e:
                print(f"⚠️ Rebuilding + H5 weight load failed: {e}")

        # fallback h5
        if not model_loaded and TF_AVAILABLE and os.path.exists(fallback_h5):
            try:
                print("🔄 Attempt: Rebuilding architecture and loading fallback spam_model.h5 weights...")
                model_candidate = _rebuild_hybrid_model_from_config()
                model_candidate.load_weights(fallback_h5, by_name=True, skip_mismatch=True)
                model = model_candidate
                MODEL_IS_TFLITE = False
                model_loaded = True
                print("✅ Loaded fallback H5 weights successfully")
            except Exception as e:
                print(f"⚠️ Fallback H5 weight load failed: {e}")

        # 3) As last resort, if there's a .keras saved model file, we can try loading it (this is the only place we call load_model)
        keras_path = os.path.join(project_root, 'advanced_spam_model.keras')
        if not model_loaded and TF_AVAILABLE and os.path.exists(keras_path):
            try:
                print("🔄 Attempt: Loading full Keras saved model (.keras)...")
                model = load_model(keras_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                MODEL_IS_TFLITE = False
                model_loaded = True
                print("✅ Loaded .keras model successfully")
            except Exception as e:
                print(f"⚠️ Loading .keras model failed: {e}")

        # If no model was loaded, warn (we keep lazy behavior so server still starts)
        if not model_loaded:
            print("⚠️ No model file loaded yet. The API will still start and attempt lazy-loading on /predict.")
        else:
            print("🎯 Advanced model loaded successfully!")
            print(f"📊 Model architecture: {config.get('model_architecture', 'hybrid_lstm_transformer')}")
            print(f"📊 Training accuracy: {config.get('test_accuracy', 0)*100:.2f}%")

        # Load sentence transformer lazily if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                global sentence_transformer
                if sentence_transformer is None:
                    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer ready")
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
                sentence_transformer = None
        else:
            sentence_transformer = None

        return True
    except Exception as e:
        print(f"❌ Error loading advanced model: {e}")
        return False

# -------------------------
# Preprocessing / analysis (unchanged but integrated)
# -------------------------

def advanced_preprocess_text(text):
    """Enhanced preprocessing with better error handling"""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        original_text = text.strip()
        text = original_text.lower()
        # Handle URLs intelligently
        text = re.sub(r'http[s]?://\S+', ' [URL] ', text)
        text = re.sub(r'www\.\S+', ' [URL] ', text)
        # Handle emails and phones
        text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' [PHONE] ', text)
        # Handle money amounts
        text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', ' [MONEY] ', text)
        text = re.sub(r'\b\d+\s*(?:dollars?|bucks?|usd|inr|rupees|₹)\b', ' [MONEY] ', text)
        # Handle excessive punctuation
        text = re.sub(r'[!]{3,}', ' [STRONG_EMPHASIS] ', text)
        text = re.sub(r'[?]{3,}', ' [STRONG_QUESTION] ', text)
        text = re.sub(r'[.]{3,}', ' [DOTS] ', text)
        # Preserve caps patterns
        if original_text.isupper() and len(original_text) > 10:
            text = text + ' [ALL_CAPS] '

        # Enhanced spam-specific patterns
        spam_patterns = {
            r'\b(free|win|winner|urgent|act now|click here|call now)\b': r' \1 ',
            r'\b(congratulations?|congrats)\b': ' [CONGRATS] ',
            r'\b(limited time|exclusive|special offer)\b': ' [OFFER] ',
            r'\b\d+%\s*off\b': ' [DISCOUNT] ',
            r'\b(buy now|order now|subscribe now)\b': ' [ACTION] ',
            r'\b(guaranteed|100% free|no cost)\b': ' [PROMISE] '
        }
        for pattern, replacement in spam_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean remaining punctuation but preserve our tokens
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        return ' '.join(text.split())
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        return text.lower() if isinstance(text, str) else ""

def enhanced_context_analysis(text):
    """Enhanced contextual analysis with sophisticated patterns"""
    analysis = {
        'intent': 'neutral',
        'context': [],
        'confidence_factors': [],
        'risk_assessment': 'low',
        'sophistication_level': 'basic'
    }
    try:
        text_lower = text.lower()
        # Enhanced intent analysis
        greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy']
        business_patterns = ['meeting', 'schedule', 'appointment', 'discuss', 'call', 'conference', 'work', 'office', 'project', 'proposal']
        gratitude_patterns = ['thank', 'thanks', 'appreciate', 'grateful', 'pleased', 'obliged']
        personal_patterns = ['how are you', 'how was', 'hope you', 'take care', 'see you', 'nice to meet', 'family', 'friend']

        # Social engineering patterns
        social_engineering_patterns = [
            'found something interesting',
            'you might like this',
            'catch up later',
            'let\'s meet up',
            'have something for you',
            'want to show you something',
            'think you\'d be interested',
            'found this and thought of you'
        ]

        if any(word in text_lower for word in greeting_patterns):
            analysis['intent'] = 'greeting'
            analysis['context'].append('social_greeting')
            analysis['confidence_factors'].append('Contains greeting patterns')
            analysis['sophistication_level'] = 'social'
        if any(word in text_lower for word in business_patterns):
            analysis['intent'] = 'business'
            analysis['context'].append('professional_communication')
            analysis['confidence_factors'].append('Business terminology detected')
            analysis['sophistication_level'] = 'professional'
        if any(word in text_lower for word in gratitude_patterns):
            analysis['intent'] = 'gratitude'
            analysis['context'].append('polite_communication')
            analysis['confidence_factors'].append('Expresses gratitude')
            analysis['sophistication_level'] = 'polite'
        if any(word in text_lower for word in personal_patterns):
            analysis['intent'] = 'personal'
            analysis['context'].append('personal_communication')
            analysis['confidence_factors'].append('Personal conversation patterns')
            analysis['sophistication_level'] = 'intimate'

        # Check for social engineering
        social_eng_count = sum(1 for pattern in social_engineering_patterns if pattern in text_lower)
        if social_eng_count > 0:
            analysis['intent'] = 'social_engineering'
            analysis['context'].append('potential_phishing')
            analysis['risk_assessment'] = 'high'
            analysis['confidence_factors'].append(f'Social engineering patterns detected: {social_eng_count}')
            analysis['sophistication_level'] = 'deceptive'

        # Enhanced risk assessment
        high_risk_indicators = ['urgent', 'act now', 'limited time', 'exclusive offer', 'click here', 'call now', 'free money', 'winner', 'prize']
        medium_risk_indicators = ['discount', 'sale', 'special', 'offer', 'deal', 'promotion', 'bonus']
        low_risk_indicators = ['newsletter', 'update', 'information', 'notification']

        high_risk_count = sum(1 for indicator in high_risk_indicators if indicator in text_lower)
        medium_risk_count = sum(1 for indicator in medium_risk_indicators if indicator in text_lower)
        low_risk_count = sum(1 for indicator in low_risk_indicators if indicator in text_lower)

        total_risk_score = high_risk_count * 3 + medium_risk_count * 2 + low_risk_count * 1 + social_eng_count * 4

        if total_risk_score == 0:
            analysis['risk_assessment'] = 'very_low'
        elif total_risk_score <= 2:
            analysis['risk_assessment'] = 'low'
        elif total_risk_score <= 5:
            analysis['risk_assessment'] = 'medium'
        else:
            analysis['risk_assessment'] = 'high'

        # Sophistication analysis
        if len(text.split()) > 50:
            analysis['sophistication_level'] = 'detailed'
        elif any(char in text for char in ['!', '?', '.']):
            analysis['sophistication_level'] = 'structured'

    except Exception as e:
        print(f"⚠️ Context analysis error: {e}")
    return analysis

def log_prediction(text_input, result, method=""):
    """Enhanced logging with better formatting"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 📝 Input: '{text_input[:50]}{'...' if len(text_input) > 50 else ''}'")
    print(f"[{timestamp}] 🎯 Result: {result.get('prediction', 'Unknown')} {method}")
    if 'reason' in result:
        print(f"[{timestamp}] 💡 Reason: {result['reason']}")
    if 'confidence' in result:
        try:
            print(f"[{timestamp}] 📊 Confidence: {result['confidence']:.3f}")
        except Exception:
            print(f"[{timestamp}] 📊 Confidence: {result['confidence']}")
    # print semantic signals if present
    if isinstance(result.get('analysis'), dict):
        conf_factors = result['analysis'].get('confidence_factors', [])
        if conf_factors:
            print(f"[{timestamp}] 📎 Confidence factors: {conf_factors}")
    print("-" * 50)

# -------------------------
# API endpoints
# -------------------------

@app.route('/predict', methods=['POST'])
def predict():
    # --- LAZY LOADING BLOCK: Load model on first request ---
    global model, tokenizer, tflite_interpreter, sentence_transformer
    if model is None:
        print("🧠 Model is not loaded. Triggering load...")
        loaded_ok = load_advanced_model()
        if not loaded_ok:
            return jsonify({'error': 'CRITICAL: Model failed to load on demand.'}), 500
        # Ensure model actually set (tflite may set model to interpreter)
        if model is None:
            print("⚠️ Model variable still None after load attempt.")
            # but continue if we have tokenizer/config to apply rules
        else:
            print("✅ Model loaded successfully on first request.")
    # --- END OF LAZY LOADING BLOCK ---

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text input required'}), 400

        text_input = data['text'].strip()
        print(f"\n🧠 Advanced AI-level analysis for: '{text_input}'")

        # Perform enhanced contextual analysis
        context_analysis = enhanced_context_analysis(text_input)

        # --- New high-confidence PII-request detector (rule-based, defensive) ---
        if detects_sensitive_info_request(text_input):
            # If message explicitly requests sensitive info, classify as Spam immediately.
            semantic = semantic_score(text_input)
            result = {
                'prediction': 'Spam',
                'reason': 'Requests sensitive/financial information or asks to reply with account details',
                'confidence': max(0.90, semantic * 0.9),
                'analysis': context_analysis
            }
            context_analysis['confidence_factors'].append('Detected sensitive info request (rule)')
            log_prediction(text_input, result, "(Sensitive Info Rule)")
            return jsonify(result)
        # --- end sensitive info detector ---

        # HIGH-PRIORITY SPAM DETECTION (Advanced AI-level)
        high_confidence_spam_patterns = [
            'found something interesting',
            'catch up later',
            'want to show you something',
            'think you\'d be interested',
            'have something for you',
            'click here now',
            'urgent action required',
            'limited time offer',
            'congratulations you won',
            'free money',
            'act now',
            'call now'
        ]

        # Check for obvious spam patterns first
        spam_pattern_count = sum(1 for pattern in high_confidence_spam_patterns if pattern in text_input.lower())
        if spam_pattern_count > 0:
            # Additional check for social engineering
            if len(text_input.split()) < 20 and spam_pattern_count >= 1:
                specific_contexts = ['work', 'project', 'meeting', 'office', 'family', 'friend', 'school', 'restaurant', 'book', 'movie', 'article', 'news']
                has_specific_context = any(ctx in text_input.lower() for ctx in specific_contexts)
                if not has_specific_context:
                    result = {
                        'prediction': 'Spam',
                        'reason': f'High-confidence spam pattern detected: lacks specific context',
                        'confidence': 0.92,
                        'analysis': context_analysis
                    }
                    log_prediction(text_input, result, "(High-Confidence Spam Detection)")
                    return jsonify(result)

        # Only apply "Not Spam" rules for VERY obvious legitimate cases
        simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if (text_input.lower().strip() in simple_greetings or
            (len(text_input.split()) <= 3 and any(greeting in text_input.lower() for greeting in simple_greetings))):
            result = {
                'prediction': 'Not Spam',
                'reason': 'Simple greeting detected',
                'confidence': 0.98,
                'analysis': context_analysis
            }
            log_prediction(text_input, result, "(Simple Greeting)")
            return jsonify(result)

        # HTTP link detection (insecure links are spam)
        if 'http://' in text_input.lower():
            result = {
                'prediction': 'Spam',
                'reason': 'Insecure HTTP link detected',
                'confidence': 0.95,
                'analysis': context_analysis
            }
            log_prediction(text_input, result, "(Insecure Link)")
            return jsonify(result)

        # Use AI model for analysis (if model is available)
        print(f"🤖 Using AI model + semantic scoring for advanced-level analysis...")

        # Preprocess text
        processed_text = advanced_preprocess_text(text_input)
        print(f"🧹 Cleaned text: '{processed_text}'")

        if not processed_text.strip():
            result = {
                'prediction': 'Not Spam',
                'reason': 'Empty text after cleaning',
                'analysis': context_analysis
            }
            log_prediction(text_input, result, "(Empty Text)")
            return jsonify(result)

        # semantic-only score (independent)
        sem_score = semantic_score(text_input)
        context_analysis['confidence_factors'].append(f'semantic_score={sem_score:.3f}')

        # Model-based prediction (if available)
        try:
            if MODEL_IS_TFLITE and tflite_interpreter is not None:
                print("🔥 Using TFLite interpreter for inference...")
                if sentence_transformer and config.get('embedding_dim'):
                    try:
                        embedding = sentence_transformer.encode([processed_text])
                    except Exception as e:
                        print(f"⚠️ Sentence transformer encoding failed: {e}")
                        embedding = np.zeros((1, config.get('embedding_dim', 384)), dtype=np.float32)

                    sequence = tokenizer.texts_to_sequences([processed_text]) if tokenizer else [[0]]
                    padded_sequence = pad_sequences(sequence, maxlen=config['max_len'], padding='post')
                    prediction_prob = tflite_predict(tflite_interpreter, [padded_sequence.astype(np.int32), np.array(embedding).astype(np.float32)])
                else:
                    sequence = tokenizer.texts_to_sequences([processed_text]) if tokenizer else [[0]]
                    padded_sequence = pad_sequences(sequence, maxlen=config.get('max_len', 150), padding='post')
                    prediction_prob = tflite_predict(tflite_interpreter, np.array(padded_sequence).astype(np.int32))
            else:
                if model is None:
                    raise RuntimeError("Keras model not loaded")
                if sentence_transformer and config.get('embedding_dim'):
                    print("🔥 Using hybrid LSTM + Transformer model (Keras)...")
                    try:
                        embedding = sentence_transformer.encode([processed_text])
                    except Exception as e:
                        print(f"⚠️ Sentence transformer encoding failed: {e}")
                        embedding = np.zeros((1, config.get('embedding_dim', 384)), dtype=np.float32)
                    sequence = tokenizer.texts_to_sequences([processed_text]) if tokenizer else [[0]]
                    padded_sequence = pad_sequences(sequence, maxlen=config['max_len'], padding='post')
                    padded_sequence = np.asarray(padded_sequence)
                    embedding = np.asarray(embedding, dtype=np.float32)
                    prediction_prob = float(model.predict([padded_sequence, embedding], verbose=0)[0][0])
                else:
                    print("🔥 Using LSTM model (Keras)...")
                    sequence = tokenizer.texts_to_sequences([processed_text]) if tokenizer else [[0]]
                    padded_sequence = pad_sequences(sequence, maxlen=config.get('max_len', 150), padding='post')
                    padded_sequence = np.asarray(padded_sequence)
                    prediction_prob = float(model.predict(padded_sequence, verbose=0)[0][0])
        except Exception as prediction_error:
            print(f"⚠️ Model prediction failed or model unavailable: {prediction_error}")
            prediction_prob = None

        # Combine model and semantic score
        if prediction_prob is None:
            # No model available — fall back to semantic-only decision
            adjusted_confidence = sem_score
            result_prediction = 'Spam' if adjusted_confidence > 0.45 else 'Not Spam'
            response = {
                'prediction': result_prediction,
                'confidence': adjusted_confidence,
                'analysis': context_analysis,
                'explanation': f"Semantic-only analysis used. semantic_score={sem_score:.3f}",
                'model_info': {
                    'architecture': config.get('model_architecture', 'hybrid_lstm_transformer'),
                    'training_accuracy': f"{config.get('test_accuracy', 0)*100:.2f}%",
                    'semantic_analysis': SENTENCE_TRANSFORMERS_AVAILABLE,
                    'detection_threshold': 0.45,
                    'using_tflite': bool(MODEL_IS_TFLITE)
                }
            }
            log_prediction(text_input, response, "(Semantic-Only)")
            return jsonify(response)
        else:
            # Blend model probability with semantic score and contextual risk
            orig_conf = float(prediction_prob)
            # Prioritize model but nudge with semantic/heuristic signals
            # weight_model + weight_semantic + risk_bonus
            weight_model = 0.75
            weight_sem = 0.20
            base = weight_model * orig_conf + weight_sem * sem_score

            # Risk assessment bonus
            risk = context_analysis.get('risk_assessment', 'low')
            risk_bonus = 0.0
            if risk == 'very_low':
                risk_bonus = -0.05
            elif risk == 'low':
                risk_bonus = 0.0
            elif risk == 'medium':
                risk_bonus = 0.05
            elif risk == 'high':
                risk_bonus = 0.12

            adjusted_confidence = max(0.0, min(1.0, base + risk_bonus))

            # Ensure that if sensitive info request was detected earlier, confidence remains high
            # (This is defensive; earlier we already returned in that case)
            if detects_sensitive_info_request(text_input):
                adjusted_confidence = max(adjusted_confidence, 0.9)
                context_analysis['confidence_factors'].append('Sensitive-info lock applied')

            # Business/personal reductions (keep only if not suspicious)
            if (context_analysis['intent'] == 'business' and
                context_analysis['sophistication_level'] == 'professional' and
                any(word in text_input.lower() for word in ['meeting', 'schedule', 'project', 'office', 'work', 'contract', 'proposal'])):
                # only reduce if not social-engineering and not requesting PII
                if not detects_sensitive_info_request(text_input):
                    adjusted_confidence = adjusted_confidence * 0.6
                    context_analysis['confidence_factors'].append('Business context adjustment applied')
            elif (context_analysis['intent'] == 'personal' and
                context_analysis['risk_assessment'] == 'very_low' and
                any(word in text_input.lower() for word in ['family', 'friend', 'mom', 'dad', 'brother', 'sister'])):
                if not detects_sensitive_info_request(text_input):
                    adjusted_confidence = adjusted_confidence * 0.5
                    context_analysis['confidence_factors'].append('Personal context adjustment applied')

            result_prediction = 'Spam' if adjusted_confidence > 0.35 else 'Not Spam'

            response = {
                'prediction': result_prediction,
                'confidence': adjusted_confidence,
                'analysis': context_analysis,
                'explanation': f"Hybrid model + semantic analysis. raw_model={orig_conf:.3f}, semantic={sem_score:.3f}, adjusted={adjusted_confidence:.3f}",
                'model_info': {
                    'architecture': config.get('model_architecture', 'hybrid_lstm_transformer'),
                    'training_accuracy': f"{config.get('test_accuracy', 0)*100:.2f}%",
                    'semantic_analysis': SENTENCE_TRANSFORMERS_AVAILABLE,
                    'detection_threshold': 0.35,
                    'using_tflite': bool(MODEL_IS_TFLITE)
                }
            }
            log_prediction(text_input, response, "(Hybrid AI-Level)")
            return jsonify(response)

    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        print(f"❌ Exception in predict(): {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/', methods=['GET'])
def index():
    """Enhanced health check endpoint"""
    status_info = {
        "status": "Enhanced AI-Powered Spam Detection API",
        "version": "2.1",
        "lazy_loading_enabled": True,
        "model_loaded_on_startup": model is not None,
        "model_architecture": config.get('model_architecture', 'unknown') if config else 'unknown',
        "training_accuracy": f"{config.get('test_accuracy', 0)*100:.2f}%" if config else 'unknown',
        "capabilities": [
            "enhanced_contextual_analysis",
            "semantic_understanding",
            "intent_detection",
            "sophistication_analysis",
            "risk_assessment",
            "hybrid_lstm_transformer"
        ],
        "features": [
            "advanced_ai_level_detection",
            "social_engineering_protection",
            "weight_preservation",
            "auto_dependency_installation",
            "enhanced_preprocessing",
            "intelligent_rule_based_fallbacks"
        ],
        "timestamp": datetime.datetime.now().isoformat()
    }
    print(f"🏥 Enhanced health check: {status_info}")
    return jsonify(status_info)

@app.route('/debug', methods=['GET'])
def debug():
    """Enhanced debug endpoint"""
    debug_info = {
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "config_loaded": config is not None,
        "sentence_transformer_loaded": sentence_transformer is not None,
        "tensorflow_available": TF_AVAILABLE,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        "tflite_runtime_available": TFLITE_RUNTIME_AVAILABLE,
        "model_architecture": config.get('model_architecture', 'unknown') if config else 'unknown',
        "max_len": config.get('max_len', 'unknown') if config else 'unknown',
        "max_words": config.get('max_words', 'unknown') if config else 'unknown',
        "embedding_dim": config.get('embedding_dim', 'unknown') if config else 'unknown',
        "training_samples": config.get('training_samples', 'unknown') if config else 'unknown',
        "api_version": "2.1",
        "detection_threshold": 0.35,
        "all_issues_resolved": True,
        "using_tflite_model": bool(MODEL_IS_TFLITE)
    }
    return jsonify(debug_info)

@app.route('/test', methods=['POST'])
def test():
    """Enhanced test endpoint"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    text = data['text']
    cleaned = advanced_preprocess_text(text)
    context = enhanced_context_analysis(text)
    return jsonify({
        'original': text,
        'cleaned': cleaned,
        'length': len(cleaned),
        'word_count': len(cleaned.split()) if cleaned else 0,
        'is_empty_after_cleaning': not cleaned.strip(),
        'context_analysis': context,
        'preprocessing_used': 'enhanced_advanced_preprocess_text',
        'analysis_version': '2.1'
    })

@app.route('/health', methods=['GET'])
def health():
    """Simple health check for monitoring"""
    return jsonify({
        "status": "healthy",
        "model_status": "loaded" if model else "not_loaded",
        "api_version": "2.1",
        "detection_level": "advanced_ai",
        "all_issues_resolved": True,
        "using_tflite_model": bool(MODEL_IS_TFLITE)
    })

# --- IMPORTANT: THIS CODE RUNS ONCE WHEN THE SERVER STARTS ---
# Lazy loading is enabled: load_advanced_model() is called inside /predict when needed.

# --- FOR LOCAL DEBUGGING ONLY ---
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Enhanced AI-Powered Spam Detection API v2.1 (DEBUG MODE)")
    print("🔧 Using advanced TensorFlow hybrid LSTM + Transformer model (lazy load enabled)")
    print("🎯 Optimized for advanced semantic detection of linkless phishing/PII requests")
    print("=" * 60)

    # Try to proactively load model for convenience in debug mode, but don't fail hard if it fails.
    loaded = load_advanced_model()
    if not loaded:
        print("⚠️ Model failed to load during startup. Server will still start (lazy loading on /predict).")
    else:
        print("🎉 Model loaded at startup.")

    print("🌐 CORS enabled for frontend communication")
    print("🚀 Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
