"""
Machine Learning based Roman Urdu to Urdu conversion model
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import RomanUrduPreprocessor
from utils.data_loader import DataLoader
from utils.urdu_utils import UrduTextProcessor

class MLModel:
    def __init__(self, model_type: str = "character_based"):
        """
        Initialize ML-based conversion model
        
        Args:
            model_type: "character_based" or "word_based"
        """
        self.model_type = model_type
        self.preprocessor = RomanUrduPreprocessor()
        self.urdu_processor = UrduTextProcessor()
        self.data_loader = DataLoader()
        
        # Model components
        self.vectorizer = None
        self.classifier = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.word_mappings = {}
        
        # Training data
        self.training_data = []
        self.is_trained = False
        
    def prepare_character_data(self, roman_texts: List[str], urdu_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare character-level training data"""
        X_chars = []
        y_chars = []
        
        # Build character vocabularies
        all_roman_chars = set()
        all_urdu_chars = set()
        
        for roman_text, urdu_text in zip(roman_texts, urdu_texts):
            roman_text = self.preprocessor.normalize_spelling(roman_text)
            urdu_text = self.urdu_processor.normalize_urdu_text(urdu_text)
            
            all_roman_chars.update(roman_text.lower())
            all_urdu_chars.update(urdu_text)
        
        # Create character mappings
        roman_chars = sorted(list(all_roman_chars))
        urdu_chars = sorted(list(all_urdu_chars))
        
        self.roman_char_to_idx = {char: idx for idx, char in enumerate(roman_chars)}
        self.urdu_char_to_idx = {char: idx for idx, char in enumerate(urdu_chars)}
        self.idx_to_urdu_char = {idx: char for char, idx in self.urdu_char_to_idx.items()}
        
        # Create training samples with context windows
        context_size = 3
        
        for roman_text, urdu_text in zip(roman_texts, urdu_texts):
            roman_text = self.preprocessor.normalize_spelling(roman_text)
            urdu_text = self.urdu_processor.normalize_urdu_text(urdu_text)
            
            # Align characters (simple alignment)
            min_len = min(len(roman_text), len(urdu_text))
            
            for i in range(min_len):
                # Context window features
                context_start = max(0, i - context_size)
                context_end = min(len(roman_text), i + context_size + 1)
                context = roman_text[context_start:context_end]
                
                # Pad context to fixed size
                context = context.ljust(2 * context_size + 1)[:2 * context_size + 1]
                
                # Convert to feature vector
                feature_vector = []
                for char in context:
                    char_features = [0] * len(roman_chars)
                    if char in self.roman_char_to_idx:
                        char_features[self.roman_char_to_idx[char]] = 1
                    feature_vector.extend(char_features)
                
                X_chars.append(feature_vector)
                
                # Target character
                if urdu_text[i] in self.urdu_char_to_idx:
                    y_chars.append(self.urdu_char_to_idx[urdu_text[i]])
                else:
                    y_chars.append(0)  # Unknown character
        
        return np.array(X_chars), np.array(y_chars)
    
    def prepare_word_data(self, roman_texts: List[str], urdu_texts: List[str]) -> Tuple[List[str], List[str]]:
        """Prepare word-level training data"""
        X_words = []
        y_words = []
        
        for roman_text, urdu_text in zip(roman_texts, urdu_texts):
            roman_words = self.preprocessor.tokenize(roman_text)
            urdu_words = self.urdu_processor.tokenize_urdu(urdu_text)
            
            # Simple word alignment
            min_len = min(len(roman_words), len(urdu_words))
            
            for i in range(min_len):
                roman_word = self.preprocessor.normalize_spelling(roman_words[i])
                urdu_word = urdu_words[i]
                
                X_words.append(roman_word)
                y_words.append(urdu_word)
                
                # Store mapping for later use
                self.word_mappings[roman_word] = urdu_word
        
        return X_words, y_words
    
    def train_character_model(self, X: np.ndarray, y: np.ndarray):
        """Train character-level model"""
        print("Training character-level model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Character-level accuracy: {accuracy:.4f}")
        
        self.is_trained = True
    
    def train_word_model(self, X: List[str], y: List[str]):
        """Train word-level model"""
        print("Training word-level model...")
        
        # Create features using character n-grams
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            max_features=5000
        )
        
        X_features = self.vectorizer.fit_transform(X)
        
        # Create unique labels
        unique_words = list(set(y))
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        y_encoded = [word_to_idx.get(word, 0) for word in y]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Word-level accuracy: {accuracy:.4f}")
        
        self.is_trained = True
    
    def train(self, data_source: str = "sample"):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        # Load training data
        roman_texts, urdu_texts = self.data_loader.get_parallel_data(data_source)
        
        if not roman_texts or not urdu_texts:
            print("No training data available!")
            return
        
        print(f"Loaded {len(roman_texts)} training samples")
        
        if self.model_type == "character_based":
            X, y = self.prepare_character_data(roman_texts, urdu_texts)
            self.train_character_model(X, y)
        
        elif self.model_type == "word_based":
            X, y = self.prepare_word_data(roman_texts, urdu_texts)
            self.train_word_model(X, y)
        
        else:
            raise ValueError("model_type must be 'character_based' or 'word_based'")
    
    def convert_word_ml(self, word: str) -> str:
        """Convert word using ML model"""
        if not self.is_trained:
            return word
        
        word = self.preprocessor.normalize_spelling(word.lower())
        
        if self.model_type == "word_based":
            # Check direct mapping first
            if word in self.word_mappings:
                return self.word_mappings[word]
            
            # Use vectorizer and classifier
            if self.vectorizer and self.classifier:
                try:
                    word_features = self.vectorizer.transform([word])
                    predicted_idx = self.classifier.predict(word_features)[0]
                    
                    if predicted_idx in self.idx_to_word:
                        return self.idx_to_word[predicted_idx]
                except:
                    pass
        
        elif self.model_type == "character_based":
            # Character-by-character conversion
            converted_chars = []
            context_size = 3
            
            for i, char in enumerate(word):
                context_start = max(0, i - context_size)
                context_end = min(len(word), i + context_size + 1)
                context = word[context_start:context_end]
                context = context.ljust(2 * context_size + 1)[:2 * context_size + 1]
                
                try:
                    # Convert to feature vector
                    feature_vector = []
                    for c in context:
                        char_features = [0] * len(self.roman_char_to_idx)
                        if c in self.roman_char_to_idx:
                            char_features[self.roman_char_to_idx[c]] = 1
                        feature_vector.extend(char_features)
                    
                    # Predict
                    predicted_idx = self.classifier.predict([feature_vector])[0]
                    
                    if predicted_idx in self.idx_to_urdu_char:
                        converted_chars.append(self.idx_to_urdu_char[predicted_idx])
                    else:
                        converted_chars.append(char)
                except:
                    converted_chars.append(char)
            
            return ''.join(converted_chars)
        
        return word
    
    def convert(self, text: str) -> str:
        """Convert Roman Urdu text to Urdu using ML model"""
        if not text:
            return ""
        
        if not self.is_trained:
            print("Model not trained! Please train the model first.")
            return text
        
        # Tokenize and convert each word
        words = self.preprocessor.tokenize(text)
        converted_words = []
        
        for word in words:
            converted_word = self.convert_word_ml(word)
            converted_words.append(converted_word)
        
        return ' '.join(converted_words)
    
    def convert_text(self, text: str) -> str:
        """Alias for convert method to match expected interface"""
        return self.convert(text)
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if not self.is_trained:
            print("No trained model to save!")
            return
        
        model_data = {
            'model_type': self.model_type,
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'word_mappings': self.word_mappings,
            'char_mappings': {
                'roman_char_to_idx': getattr(self, 'roman_char_to_idx', {}),
                'urdu_char_to_idx': getattr(self, 'urdu_char_to_idx', {}),
                'idx_to_urdu_char': getattr(self, 'idx_to_urdu_char', {})
            },
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            model_data = joblib.load(model_path)
            
            self.model_type = model_data['model_type']
            self.classifier = model_data['classifier']
            self.vectorizer = model_data['vectorizer']
            self.word_mappings = model_data['word_mappings']
            self.is_trained = model_data['is_trained']
            
            # Load character mappings
            char_mappings = model_data.get('char_mappings', {})
            self.roman_char_to_idx = char_mappings.get('roman_char_to_idx', {})
            self.urdu_char_to_idx = char_mappings.get('urdu_char_to_idx', {})
            self.idx_to_urdu_char = char_mappings.get('idx_to_urdu_char', {})
            
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict:
        """Evaluate model on test data"""
        if not self.is_trained:
            print("Model not trained!")
            return {}
        
        predictions = []
        references = []
        correct_words = 0
        total_words = 0
        
        for roman_text, expected_urdu in test_data:
            predicted_urdu = self.convert(roman_text)
            predictions.append(predicted_urdu)
            references.append(expected_urdu)
            
            # Word-level accuracy
            pred_words = predicted_urdu.split()
            ref_words = expected_urdu.split()
            
            for pred_word, ref_word in zip(pred_words, ref_words):
                total_words += 1
                if pred_word == ref_word:
                    correct_words += 1
        
        word_accuracy = correct_words / total_words if total_words > 0 else 0
        
        # Sentence-level accuracy
        correct_sentences = sum(1 for pred, ref in zip(predictions, references) 
                              if pred.strip() == ref.strip())
        sentence_accuracy = correct_sentences / len(test_data) if test_data else 0
        
        evaluation = {
            'word_accuracy': word_accuracy,
            'sentence_accuracy': sentence_accuracy,
            'total_words': total_words,
            'correct_words': correct_words,
            'total_sentences': len(test_data),
            'correct_sentences': correct_sentences,
            'predictions': predictions,
            'references': references
        }
        
        return evaluation

class EnsembleModel:
    """Ensemble of dictionary and ML models"""
    
    def __init__(self, dictionary_model, ml_model):
        self.dictionary_model = dictionary_model
        self.ml_model = ml_model
    
    def convert(self, text: str) -> str:
        """Convert using ensemble approach"""
        # Get predictions from both models
        dict_prediction = self.dictionary_model.convert(text)
        ml_prediction = self.ml_model.convert(text)
        
        # Simple voting: prefer dictionary if it has good coverage
        words = text.split()
        dict_words = dict_prediction.split()
        ml_words = ml_prediction.split()
        
        final_words = []
        
        for i, word in enumerate(words):
            dict_word = dict_words[i] if i < len(dict_words) else word
            ml_word = ml_words[i] if i < len(ml_words) else word
            
            # Use dictionary if word was successfully converted (contains Urdu chars)
            if (dict_word != word and 
                any(char in self.dictionary_model.urdu_processor.urdu_chars for char in dict_word)):
                final_words.append(dict_word)
            else:
                final_words.append(ml_word)
        
        return ' '.join(final_words)
    
    def convert_text(self, text: str) -> str:
        """Alias for convert method to match expected interface"""
        return self.convert(text)
        
        return ' '.join(final_words)

if __name__ == "__main__":
    # Test the ML model
    print("Testing ML Model...")
    
    # Test word-based model
    word_model = MLModel(model_type="word_based")
    word_model.train("sample")
    
    test_sentences = [
        "aap kaise hain",
        "main theek hun",
        "yeh kitab achi hai"
    ]
    
    print("\nWord-based Model Results:")
    for sentence in test_sentences:
        converted = word_model.convert(sentence)
        print(f"Roman: {sentence}")
        print(f"Urdu:  {converted}")
        print()
    
    # Test character-based model
    char_model = MLModel(model_type="character_based")
    char_model.train("sample")
    
    print("\nCharacter-based Model Results:")
    for sentence in test_sentences:
        converted = char_model.convert(sentence)
        print(f"Roman: {sentence}")
        print(f"Urdu:  {converted}")
        print()
