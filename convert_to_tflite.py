import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, Concatenate

def create_hybrid_model_for_conversion():
    """
    Recreates the exact architecture of your trained model so we can load the weights cleanly.
    This function remains the same as it correctly mirrors your train_model.py.
    """
    print("🏗️ Rebuilding the exact model architecture for conversion...")
    max_len = 150
    vocab_size = 8000
    embedding_dim = 384

    text_input = Input(shape=(max_len,), name='text_input')
    embedding_input = Input(shape=(embedding_dim,), name='embedding_input')
    embedding_layer = Embedding(vocab_size, 128)(text_input)
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
    
    model = Model(inputs=[text_input, embedding_input], outputs=output)
    print("✅ Model architecture rebuilt successfully.")
    return model

def convert_h5_to_tflite_final():
    """
    The final, robust conversion script that handles all previous errors.
    """
    model_path_h5 = 'advanced_spam_model.h5'
    model_path_tflite = 'model.tflite'

    if not os.path.exists(model_path_h5):
        print(f"❌ Error: Model file '{model_path_h5}' not found in the project root.")
        return False

    try:
        # Step 1: Create a fresh instance of your model architecture.
        model = create_hybrid_model_for_conversion()

        # Step 2: Load the saved weights into this fresh model.
        print(f"🔄 Loading weights from '{model_path_h5}'...")
        model.load_weights(model_path_h5)
        print("✅ Weights loaded successfully.")

        # Step 3: Initialize the TFLite converter.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # --- THE FINAL FIX FOR THE TENSORLISTRESERVE ERROR ---
        # As suggested by the error log, we enable a compatibility mode.
        print("🔧 Applying TFLite conversion compatibility flags...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable default TFLite ops.
            tf.lite.OpsSet.SELECT_TF_OPS    # Enable select TensorFlow ops for compatibility.
        ]
        converter._experimental_lower_tensor_list_ops = False
        # --- END OF FIX ---

        print("🔄 Converting model to TensorFlow Lite format...")
        tflite_model = converter.convert()
        print("✅ Conversion successful.")

        # Step 4: Save the new .tflite model.
        with open(model_path_tflite, 'wb') as f:
            f.write(tflite_model)

        print(f"🎉 Success! Your new '{model_path_tflite}' file has been created in the project root.")
        return True

    except Exception as e:
        print(f"❌ An error occurred during the final conversion process: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    convert_h5_to_tflite_final()
