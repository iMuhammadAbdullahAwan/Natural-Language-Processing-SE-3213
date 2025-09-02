"""
Dictionary-based Roman Urdu to Urdu conversion model
"""
import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import RomanUrduPreprocessor, expand_dictionary_with_variations
from utils.urdu_utils import UrduTextProcessor

class DictionaryModel:
    def __init__(self, dictionary_path: Optional[str] = None):
        """Initialize dictionary-based conversion model"""
        self.preprocessor = RomanUrduPreprocessor()
        self.urdu_processor = UrduTextProcessor()
        
        # Load dictionary
        if dictionary_path is None:
            dictionary_path = Path(__file__).parent.parent / "data" / "roman_urdu_dictionary.json"
        
        self.dictionary = self.load_dictionary(dictionary_path)
        self.expanded_dictionary = expand_dictionary_with_variations(self.dictionary, self.preprocessor)
        
        # Statistics
        self.conversion_stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'word_level_accuracy': 0.0,
            'unknown_words': set()
        }
    
    def load_dictionary(self, file_path: str) -> Dict[str, str]:
        """Load Roman Urdu to Urdu dictionary"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('dictionary', {})
        except FileNotFoundError:
            print(f"Dictionary file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing dictionary: {e}")
            return {}
    
    def convert_word(self, word: str) -> str:
        """Convert a single Roman Urdu word to Urdu"""
        if not word:
            return ""
        
        # Normalize the word
        normalized_word = self.preprocessor.normalize_spelling(word.lower())
        
        # First try exact match
        if normalized_word in self.expanded_dictionary:
            return self.expanded_dictionary[normalized_word]
        
        # Try original word
        if word.lower() in self.expanded_dictionary:
            return self.expanded_dictionary[word.lower()]
        
        # Try without normalization
        if word in self.expanded_dictionary:
            return self.expanded_dictionary[word]
        
        # Try fuzzy matching for close variations
        fuzzy_match = self.fuzzy_match(normalized_word)
        if fuzzy_match:
            return fuzzy_match
        
        # If no match found, return original word
        self.conversion_stats['unknown_words'].add(word)
        return word
    
    def fuzzy_match(self, word: str, threshold: float = 0.8) -> Optional[str]:
        """Find fuzzy matches for words not in dictionary"""
        best_match = None
        best_score = 0
        
        for dict_word in self.expanded_dictionary.keys():
            score = self.calculate_similarity(word, dict_word)
            if score > threshold and score > best_score:
                best_score = score
                best_match = self.expanded_dictionary[dict_word]
        
        return best_match
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using edit distance"""
        if not word1 or not word2:
            return 0.0
        
        # Simple character overlap metric
        chars1 = set(word1.lower())
        chars2 = set(word2.lower())
        
        intersection = chars1.intersection(chars2)
        union = chars1.union(chars2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def convert_sentence(self, sentence: str) -> str:
        """Convert a Roman Urdu sentence to Urdu"""
        if not sentence:
            return ""
        
        # Preprocess the sentence
        preprocessed = self.preprocessor.normalize_spelling(sentence)
        words = self.preprocessor.tokenize(preprocessed)
        
        # Convert each word
        urdu_words = []
        successful_words = 0
        
        for word in words:
            urdu_word = self.convert_word(word)
            urdu_words.append(urdu_word)
            
            # Check if conversion was successful (word changed)
            if urdu_word != word and urdu_word in self.urdu_processor.urdu_chars:
                successful_words += 1
        
        # Update statistics
        self.conversion_stats['total_conversions'] += 1
        if successful_words > 0:
            self.conversion_stats['successful_conversions'] += 1
        
        word_accuracy = successful_words / len(words) if words else 0
        self.conversion_stats['word_level_accuracy'] = (
            (self.conversion_stats['word_level_accuracy'] * (self.conversion_stats['total_conversions'] - 1) + word_accuracy) /
            self.conversion_stats['total_conversions']
        )
        
        return ' '.join(urdu_words)
    
    def convert(self, text: str) -> str:
        """Main conversion method"""
        if not text:
            return ""
        
        # Handle multiple sentences
        sentences = re.split(r'[.!?]+', text)
        converted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                converted = self.convert_sentence(sentence)
                converted_sentences.append(converted)
        
        return '۔ '.join(converted_sentences) + ('۔' if converted_sentences else '')
    
    def convert_text(self, text: str) -> str:
        """Alias for convert method to match expected interface"""
        return self.convert(text)
    
    def batch_convert(self, texts: List[str]) -> List[str]:
        """Convert multiple texts"""
        return [self.convert(text) for text in texts]
    
    def add_to_dictionary(self, roman_word: str, urdu_word: str):
        """Add new word pair to dictionary"""
        self.dictionary[roman_word.lower()] = urdu_word
        self.expanded_dictionary[roman_word.lower()] = urdu_word
        
        # Add variations
        variations = self.preprocessor.generate_variations(roman_word.lower())
        for variation in variations:
            if variation not in self.expanded_dictionary:
                self.expanded_dictionary[variation] = urdu_word
    
    def save_dictionary(self, file_path: str):
        """Save updated dictionary"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({'dictionary': self.dictionary}, f, ensure_ascii=False, indent=2)
            print(f"Dictionary saved to: {file_path}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")
    
    def get_statistics(self) -> Dict:
        """Get conversion statistics"""
        stats = self.conversion_stats.copy()
        stats['dictionary_size'] = len(self.dictionary)
        stats['expanded_dictionary_size'] = len(self.expanded_dictionary)
        stats['unknown_words_count'] = len(stats['unknown_words'])
        stats['unknown_words'] = list(stats['unknown_words'])
        return stats
    
    def get_coverage(self, text_list: List[str]) -> Dict:
        """Calculate dictionary coverage for given texts"""
        total_words = 0
        covered_words = 0
        unknown_words = set()
        
        for text in text_list:
            words = self.preprocessor.tokenize(text)
            total_words += len(words)
            
            for word in words:
                normalized_word = self.preprocessor.normalize_spelling(word.lower())
                if normalized_word in self.expanded_dictionary:
                    covered_words += 1
                else:
                    unknown_words.add(word)
        
        coverage = {
            'total_words': total_words,
            'covered_words': covered_words,
            'coverage_percentage': (covered_words / total_words * 100) if total_words > 0 else 0,
            'unknown_words': list(unknown_words),
            'unknown_words_count': len(unknown_words)
        }
        
        return coverage
    
    def suggest_additions(self, text_list: List[str], min_frequency: int = 2) -> List[str]:
        """Suggest words to add to dictionary based on frequency"""
        word_freq = {}
        
        for text in text_list:
            words = self.preprocessor.tokenize(text)
            for word in words:
                normalized_word = self.preprocessor.normalize_spelling(word.lower())
                if normalized_word not in self.expanded_dictionary:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return words that appear frequently
        suggestions = [word for word, freq in word_freq.items() if freq >= min_frequency]
        suggestions.sort(key=lambda x: word_freq[x], reverse=True)
        
        return suggestions
    
    def interactive_learning(self, roman_word: str, urdu_word: str):
        """Interactive learning mode to add new words"""
        self.add_to_dictionary(roman_word, urdu_word)
        print(f"Added: {roman_word} -> {urdu_word}")
    
    def evaluate_on_data(self, test_data: List[Tuple[str, str]]) -> Dict:
        """Evaluate model on test data"""
        total_samples = len(test_data)
        correct_conversions = 0
        word_level_correct = 0
        total_words = 0
        
        predictions = []
        references = []
        
        for roman_text, expected_urdu in test_data:
            predicted_urdu = self.convert(roman_text)
            predictions.append(predicted_urdu)
            references.append(expected_urdu)
            
            # Sentence level accuracy
            if predicted_urdu.strip() == expected_urdu.strip():
                correct_conversions += 1
            
            # Word level accuracy
            pred_words = predicted_urdu.split()
            ref_words = expected_urdu.split()
            
            for pred_word, ref_word in zip(pred_words, ref_words):
                total_words += 1
                if pred_word == ref_word:
                    word_level_correct += 1
        
        evaluation = {
            'sentence_accuracy': correct_conversions / total_samples if total_samples > 0 else 0,
            'word_accuracy': word_level_correct / total_words if total_words > 0 else 0,
            'total_samples': total_samples,
            'correct_sentences': correct_conversions,
            'total_words': total_words,
            'correct_words': word_level_correct,
            'predictions': predictions,
            'references': references
        }
        
        return evaluation

if __name__ == "__main__":
    # Test the dictionary model
    model = DictionaryModel()
    
    # Test single word conversion
    test_words = ["aap", "kaise", "hain", "main", "theek", "hun"]
    print("Word conversions:")
    for word in test_words:
        converted = model.convert_word(word)
        print(f"{word} -> {converted}")
    
    print("\n" + "="*50 + "\n")
    
    # Test sentence conversion
    test_sentences = [
        "aap kaise hain",
        "main theek hun",
        "aap ka naam kya hai",
        "yeh kitab bht achi hai"
    ]
    
    print("Sentence conversions:")
    for sentence in test_sentences:
        converted = model.convert(sentence)
        print(f"Roman: {sentence}")
        print(f"Urdu:  {converted}")
        print()
    
    # Display statistics
    print("Model Statistics:")
    stats = model.get_statistics()
    for key, value in stats.items():
        if key != 'unknown_words' or len(value) < 10:  # Don't print long lists
            print(f"{key}: {value}")
    
    # Test coverage
    coverage = model.get_coverage(test_sentences)
    print(f"\nDictionary Coverage: {coverage['coverage_percentage']:.2f}%")
