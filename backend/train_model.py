import pandas as pd
import numpy as np
import re
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Try different TensorFlow import methods
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D, Concatenate
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    print("✅ TensorFlow/Keras imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    try:
        import tf_keras as keras
        from tf_keras.models import Model, Sequential
        from tf_keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D, Concatenate
        from tf_keras.preprocessing.text import Tokenizer
        from tf_keras.preprocessing.sequence import pad_sequences
        from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tf_keras.optimizers import Adam
        print("✅ tf-keras imported as fallback")
    except ImportError as e2:
        print(f"❌ tf-keras import error: {e2}")
        print("Please reinstall TensorFlow: pip install tensorflow==2.15.0")
        exit(1)

# Try importing sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  Sentence Transformers not available")

class AdvancedSpamDetector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        self.config = {}
        
    def advanced_preprocess_text(self, text):
        """Enhanced text preprocessing with stopwords removal"""
        if not isinstance(text, str) or not text.strip():
            return ""
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
        text = re.sub(r'\b\d+\s*(?:dollars?|bucks?|USD)\b', ' [MONEY] ', text)
        # Handle excessive punctuation
        text = re.sub(r'[!]{3,}', ' [STRONG_EMPHASIS] ', text)
        text = re.sub(r'[?]{3,}', ' [STRONG_QUESTION] ', text)
        text = re.sub(r'[.]{3,}', ' [DOTS] ', text)
        # Preserve caps patterns
        if original_text.isupper() and len(original_text) > 10:
            text = text + ' [ALL_CAPS] '
        # Spam-specific patterns
        spam_patterns = {
            r'\b(free|win|winner|urgent|act now|click here|call now)\b': r' \1 ',
            r'\b(congratulations?|congrats)\b': ' [CONGRATS] ',
            r'\b(limited time|exclusive|special offer)\b': ' [OFFER] ',
            r'\b\d+%\s*off\b': ' [DISCOUNT] '
        }
        for pattern, replacement in spam_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Clean remaining punctuation
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in STOPWORDS])
        return text
    
    def create_contextual_features(self, texts, batch_size=32):
        """Create semantic embeddings with fallback"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️  Using dummy embeddings - reduced accuracy but will work")
            return np.random.rand(len(texts), 384)
        
        print("🧠 Creating contextual embeddings...")
        
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.sentence_transformer.encode(
                    batch_texts, 
                    show_progress_bar=True,
                    batch_size=16,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            embeddings = np.vstack(all_embeddings)
            print(f"✅ Created embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return np.random.rand(len(texts), 384)
    
    def create_hybrid_model(self, embedding_dim=384, max_len=150, vocab_size=8000):
        """Create hybrid model with better error handling"""
        print("🏗️ Building advanced hybrid model...")
        
        try:
            # Input layers
            text_input = Input(shape=(max_len,), name='text_input')
            embedding_input = Input(shape=(embedding_dim,), name='embedding_input')
            
            # Traditional text processing branch
            embedding_layer = Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)(text_input)
            lstm_layer = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding_layer)
            lstm_pool = GlobalMaxPooling1D()(lstm_layer)
            
            # Semantic understanding branch
            semantic_dense1 = Dense(256, activation='relu')(embedding_input)
            semantic_dropout1 = Dropout(0.3)(semantic_dense1)
            semantic_dense2 = Dense(128, activation='relu')(semantic_dropout1)
            semantic_dropout2 = Dropout(0.2)(semantic_dense2)
            
            # Combine both approaches
            merged = Concatenate()([lstm_pool, semantic_dropout2])
            
            # Classification head
            dense1 = Dense(128, activation='relu')(merged)
            dropout1 = Dropout(0.4)(dense1)
            dense2 = Dense(64, activation='relu')(dropout1)
            dropout2 = Dropout(0.3)(dense2)
            dense3 = Dense(32, activation='relu')(dropout2)
            dropout3 = Dropout(0.2)(dense3)
            
            output = Dense(1, activation='sigmoid', name='spam_prediction')(dropout3)
            model = Model(inputs=[text_input, embedding_input], outputs=output)
            
            return model
            
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            print("Falling back to simple LSTM model...")
            
            # Fallback to simple model
            model = Sequential([
                Embedding(vocab_size, 128, input_length=max_len),
                LSTM(64, dropout=0.3, recurrent_dropout=0.3),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            return model

def train_advanced_model():
    """Main training function with better error handling"""
    print("🚀 Starting Advanced Spam Detection Training...")
    print("=" * 60)
    
    start_time = time.time()
    detector = AdvancedSpamDetector()
    
    try:
        # Load dataset
        print("📊 Loading dataset...")
        if not os.path.exists('enron_spam_data.csv'):
            raise FileNotFoundError("enron_spam_data.csv not found!")
        
        df = pd.read_csv('enron_spam_data.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} total emails")
        
        # Data validation and cleaning
        if 'Message' not in df.columns or 'Spam/Ham' not in df.columns:
            raise ValueError("Required columns not found!")
        
        initial_count = len(df)
        df.dropna(subset=['Message'], inplace=True)
        df = df[df['Message'].str.strip().astype(bool)]
        print(f"📉 Removed {initial_count - len(df)} invalid messages")
        
        # Check label distribution
        label_dist = df['Spam/Ham'].value_counts()
        print(f"📊 Label distribution: {dict(label_dist)}")
        
        # Advanced preprocessing
        print("🧹 Advanced preprocessing...")
        df['Processed_Message'] = df['Message'].apply(detector.advanced_preprocess_text)
        df = df[df['Processed_Message'].str.len() > 3]
        
        texts = df['Processed_Message'].tolist()
        labels = df['Spam/Ham'].map({'ham': 0, 'spam': 1}).values
        
        print(f"📈 Final dataset: {len(texts)} emails")
        
        # Create contextual embeddings
        embeddings = detector.create_contextual_features(texts)
        
        # Split data
        print("🔄 Splitting data...")
        X_train_text, X_test_text, X_train_emb, X_test_emb, y_train, y_test = train_test_split(
            texts, embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        # Tokenization
        print("🔤 Creating tokenizer...")
        max_words = 8000
        max_len = 150
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train_text)
        X_train_seq = tokenizer.texts_to_sequences(X_train_text)
        X_test_seq = tokenizer.texts_to_sequences(X_test_text)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
        print(f"📊 Vocabulary size: {len(tokenizer.word_index)}")
        # Handle class imbalance with RandomOverSampler
        print("⚖️  Applying RandomOverSampler for class imbalance...")
        ros = RandomOverSampler(random_state=42)
        X_train_pad_ros, y_train_ros = ros.fit_resample(X_train_pad, y_train)
        X_train_emb_ros, _ = ros.fit_resample(X_train_emb, y_train)
        
        # Create and compile model
        model = detector.create_hybrid_model(
            embedding_dim=embeddings.shape[1],
            max_len=max_len,
            vocab_size=max_words
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("📋 Model Architecture:")
        model.summary()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
        ]
        
        # Train model
        print("🎯 Training advanced model...")
        training_start = time.time()
        # Check if we have hybrid model or simple model
        if len(model.input_shape) == 2 or hasattr(model, 'inputs'):
            # Hybrid model with two inputs
            history = model.fit(
                [X_train_pad_ros, X_train_emb_ros], y_train_ros,
                validation_data=([X_test_pad, X_test_emb], y_test),
                epochs=20,  # Increased epochs for better learning
                batch_size=64,  # Increased batch size
                callbacks=callbacks,
                verbose=1
            )
            test_results = model.evaluate([X_test_pad, X_test_emb], y_test, verbose=0)
        else:
            # Simple model with one input
            history = model.fit(
                X_train_pad_ros, y_train_ros,
                validation_data=(X_test_pad, y_test),
                epochs=20,
                batch_size=64,
                callbacks=callbacks,
                verbose=1
            )
            test_results = model.evaluate(X_test_pad, y_test, verbose=0)
        
        training_time = time.time() - training_start
        
        # Evaluate model
        print("\n📊 Model Evaluation:")
        test_loss, test_acc = test_results[0], test_results[1]
        print(f"✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"📊 Test Loss: {test_loss:.4f}")
        # Additional metrics
        if len(model.input_shape) == 2 or hasattr(model, 'inputs'):
            y_pred = (model.predict([X_test_pad, X_test_emb]) > 0.5).astype(int)
        else:
            y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        
        # Save everything
        print("\n💾 Saving model...")
        
        model.save('advanced_spam_model.h5')
        print("✅ Advanced model saved")
        
        if detector.sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                detector.sentence_transformer.save('sentence_transformer_model')
                print("✅ Sentence transformer saved")
            except Exception as e:
                print(f"⚠️  Could not save sentence transformer: {e}")
        
        with open('advanced_tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
        print("✅ Tokenizer saved")
        
        config = {
            'max_words': max_words,
            'max_len': max_len,
            'embedding_dim': embeddings.shape[1],
            'model_architecture': 'hybrid_lstm_transformer',
            'training_samples': len(X_train_text),
            'test_accuracy': float(test_acc)
        }
        
        with open('advanced_model_config.pickle', 'wb') as f:
            pickle.dump(config, f)
        print("✅ Configuration saved")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Advanced Training Complete!")
        print("=" * 60)
        print(f"🎯 Final Accuracy: {test_acc*100:.2f}%")
        print(f"⏱️  Training Time: {training_time/60:.2f} minutes")
        print(f"⏱️  Total Time: {total_time/60:.2f} minutes")
        
        return detector, model, tokenizer, config
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    print("🤖 Advanced ChatGPT-Level Spam Detection Training")
    print("🔧 With improved error handling and compatibility")
    print("\n" + "=" * 60)
    
    result = train_advanced_model()
    
    if result[0] is not None:
        print("\n✅ Training completed successfully!")
        print("🚀 Ready to use with enhanced API!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
