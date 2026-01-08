"""
LSTM-based Text Generation using TensorFlow and Keras
A production-ready implementation for generating text using LSTM neural networks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import re
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTextGenerator:
    def __init__(self, sequence_length=50, embedding_dim=100, lstm_units=128):
        """
        Initialize LSTM Text Generator
        
        Args:
            sequence_length (int): Length of input sequences
            embedding_dim (int): Dimension of embedding layer
            lstm_units (int): Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess text data from file
        
        Args:
            file_path (str): Path to text file
            
        Returns:
            str: Preprocessed text
        """
        print("Loading and preprocessing data...")
        
        # Load text data
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        print(f"Text length after preprocessing: {len(text)} characters")
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text using Keras Tokenizer
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            list: Tokenized sequences
        """
        print("Tokenizing text...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text])
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences([text])[0]
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total sequences: {len(sequences)}")
        
        return sequences
    
    def create_sequences(self, sequences):
        """
        Create input-output sequence pairs
        
        Args:
            sequences (list): Tokenized sequences
            
        Returns:
            tuple: (X, y) input and output sequences
        """
        print("Creating input-output sequences...")
        
        X, y = [], []
        
        # Create sequences of specified length
        for i in range(self.sequence_length, len(sequences)):
            X.append(sequences[i-self.sequence_length:i])
            y.append(sequences[i])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Pad sequences to ensure uniform length
        X = pad_sequences(X, maxlen=self.sequence_length, padding='pre')
        
        # Convert outputs to categorical (one-hot encoding)
        y = to_categorical(y, num_classes=self.vocab_size)
        
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        return X, y
    
    def build_model(self):
        """
        Build LSTM model architecture
        
        Returns:
            Sequential: Compiled Keras model
        """
        print("Building LSTM model...")
        
        model = Sequential([
            # Embedding layer to convert tokens to dense vectors
            Embedding(input_dim=self.vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.sequence_length),
            
            # First LSTM layer with return sequences for stacking
            LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            
            # Second LSTM layer for deeper learning
            LSTM(self.lstm_units, dropout=0.2, recurrent_dropout=0.2),
            
            # Dropout for regularization
            Dropout(0.3),
            
            # Dense output layer with softmax for probability distribution
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X, y, epochs=50, batch_size=128, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            X (np.array): Input sequences
            y (np.array): Output sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
        """
        print("Training model...")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def generate_text(self, seed_text, num_words=50, temperature=1.0):
        """
        Generate text using trained model
        
        Args:
            seed_text (str): Initial text to start generation
            num_words (int): Number of words to generate
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            str: Generated text
        """
        # Preprocess seed text
        seed_text = seed_text.lower()
        seed_text = re.sub(r'[^a-z\s]', '', seed_text)
        
        # Convert seed text to sequence
        seed_sequence = self.tokenizer.texts_to_sequences([seed_text])[0]
        
        # Ensure seed sequence has minimum length
        if len(seed_sequence) < self.sequence_length:
            # Pad with zeros if too short
            seed_sequence = [0] * (self.sequence_length - len(seed_sequence)) + seed_sequence
        else:
            # Take last sequence_length tokens
            seed_sequence = seed_sequence[-self.sequence_length:]
        
        generated_text = seed_text
        
        # Generate words iteratively
        for _ in range(num_words):
            # Prepare input sequence
            input_sequence = np.array([seed_sequence])
            input_sequence = pad_sequences(input_sequence, maxlen=self.sequence_length, padding='pre')
            
            # Predict next word probabilities
            predictions = self.model.predict(input_sequence, verbose=0)[0]
            
            # Apply temperature sampling
            predictions = np.log(predictions + 1e-8) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions)
            
            # Sample next word index
            next_word_index = np.random.choice(len(predictions), p=predictions)
            
            # Convert index back to word
            next_word = None
            for word, index in self.tokenizer.word_index.items():
                if index == next_word_index:
                    next_word = word
                    break
            
            if next_word:
                generated_text += " " + next_word
                # Update seed sequence for next prediction
                seed_sequence = seed_sequence[1:] + [next_word_index]
            else:
                break
        
        return generated_text
    
    def save_model_and_tokenizer(self, model_path='lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save trained model and tokenizer"""
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model_and_tokenizer(self, model_path='lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load trained model and tokenizer"""
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Model and tokenizer loaded successfully!")


def create_sample_dataset(file_path='sample_text.txt'):
    """
    Create a sample text dataset if none exists
    """
    sample_text = """
    To be or not to be that is the question whether tis nobler in the mind to suffer
    the slings and arrows of outrageous fortune or to take arms against a sea of troubles
    and by opposing end them to die to sleep no more and by a sleep to say we end
    the heartache and the thousand natural shocks that flesh is heir to tis a consummation
    devoutly to be wished to die to sleep to sleep perchance to dream ay theres the rub
    for in that sleep of death what dreams may come when we have shuffled off this mortal coil
    must give us pause theres the respect that makes calamity of so long life
    for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely
    the pangs of despised love the laws delay the insolence of office and the spurns
    that patient merit of the unworthy takes when he himself might his quietus make
    with a bare bodkin who would fardels bear to grunt and sweat under a weary life
    but that the dread of something after death the undiscovered country from whose bourn
    no traveler returns puzzles the will and makes us rather bear those ills we have
    than fly to others that we know not of thus conscience does make cowards of us all
    and thus the native hue of resolution is sicklied oer with the pale cast of thought
    and enterprises of great pith and moment with this regard their currents turn awry
    and lose the name of action
    """ * 10  # Repeat to have more training data
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    print(f"Sample dataset created at {file_path}")


def main():
    """
    Main function to demonstrate LSTM text generation
    """
    print("=== LSTM Text Generation Project ===\n")
    
    # Configuration parameters
    SEQUENCE_LENGTH = 30  # Shorter for demo, can be increased for better results
    EMBEDDING_DIM = 100
    LSTM_UNITS = 128
    EPOCHS = 20  # Reduced for demo, increase for production
    
    # Create sample dataset if it doesn't exist
    dataset_path = 'sample_text.txt'
    if not os.path.exists(dataset_path):
        create_sample_dataset(dataset_path)
    
    # Initialize text generator
    generator = LSTMTextGenerator(
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )
    
    try:
        # Step 1: Load and preprocess data
        text = generator.load_and_preprocess_data(dataset_path)
        
        # Step 2: Tokenize text
        sequences = generator.tokenize_text(text)
        
        # Step 3: Create input-output sequences
        X, y = generator.create_sequences(sequences)
        
        # Step 4: Build model
        model = generator.build_model()
        
        # Step 5: Train model
        history = generator.train_model(X, y, epochs=EPOCHS)
        
        # Step 6: Save model and tokenizer
        generator.save_model_and_tokenizer()
        
        # Step 7: Generate text with different seed inputs
        print("\n=== Text Generation Results ===\n")
        
        seed_texts = [
            "to be or not to be",
            "the question whether tis nobler"
        ]
        
        for i, seed in enumerate(seed_texts, 1):
            print(f"Seed {i}: '{seed}'")
            generated = generator.generate_text(seed, num_words=30, temperature=0.8)
            print(f"Generated: {generated}\n")
            print("-" * 80 + "\n")
        
        # Bonus: Experiment with different parameters
        print("=== Bonus: Parameter Experiments ===\n")
        
        # Different temperature values
        print("Temperature experiment (same seed, different randomness):")
        seed = "to be or not"
        for temp in [0.5, 1.0, 1.5]:
            generated = generator.generate_text(seed, num_words=20, temperature=temp)
            print(f"Temperature {temp}: {generated}")
        
        print("\n" + "=" * 80)
        print("Experiment Notes:")
        print("- Longer sequences (50-100) capture more context but need more data")
        print("- Deeper LSTM layers (3-4) learn complex patterns but risk overfitting")
        print("- Lower temperature (0.5) = more conservative, higher (1.5) = more creative")
        print("- More training data and epochs significantly improve quality")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Make sure you have sufficient training data and computational resources.")


if __name__ == "__main__":
    main()