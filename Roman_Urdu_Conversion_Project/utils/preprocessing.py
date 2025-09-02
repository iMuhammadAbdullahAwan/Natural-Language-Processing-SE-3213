"""
Preprocessing utilities for Roman Urdu to Urdu conversion
"""
import re
import json
import string
from typing import List, Dict, Tuple

class RomanUrduPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with common spelling variations"""
        self.spelling_variations = {
            'k': ['c', 'ck'],
            's': ['c', 'z'],
            'z': ['s'],
            'f': ['ph'],
            'w': ['v'],
            'i': ['e', 'ee'],
            'u': ['o', 'oo'],
            'a': ['aa', 'e'],
            'y': ['i', 'ee'],
            'h': [''],  # silent h
        }
        
        # Common word normalization patterns
        self.normalization_patterns = {
            r'\bap\b': 'aap',
            r'\bho\b': 'hai',
            r'\bhe\b': 'hai',
            r'\bhy\b': 'hai',
            r'\bhen\b': 'hain',
            r'\bhyn\b': 'hain',
            r'\bme\b': 'main',
            r'\bmein\b': 'main',
            r'\bye\b': 'yeh',
            r'\bya\b': 'yeh',
            r'\bwo\b': 'woh',
            r'\bo\b': 'woh',
            r'\bkesy\b': 'kaise',
            r'\bkesay\b': 'kaise',
            r'\bkese\b': 'kaise',
            r'\bkyu\b': 'kyun',
            r'\bkiun\b': 'kyun',
            r'\bfir\b': 'phir',
            r'\bfer\b': 'phir',
            r'\bgar\b': 'ghar',
            r'\bpani\b': 'paani',
            r'\bkana\b': 'khana',
            r'\bachha\b': 'acha',
            r'\bachchha\b': 'acha',
            r'\bbra\b': 'bura',
            r'\bbda\b': 'bara',
            r'\bchta\b': 'chota',
            r'\bchhota\b': 'chota',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize the input text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        
        return text
    
    def normalize_spelling(self, text: str) -> str:
        """Normalize common spelling variations"""
        text = self.clean_text(text)
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = self.clean_text(text)
        return text.split()
    
    def generate_variations(self, word: str) -> List[str]:
        """Generate possible spelling variations of a word"""
        variations = [word]
        
        # Simple character-level variations
        for char, replacements in self.spelling_variations.items():
            if char in word:
                for replacement in replacements:
                    variation = word.replace(char, replacement)
                    if variation != word and variation not in variations:
                        variations.append(variation)
        
        return variations
    
    def preprocess_for_training(self, roman_text: str, urdu_text: str) -> Tuple[str, str]:
        """Preprocess training data pairs"""
        roman_cleaned = self.normalize_spelling(roman_text)
        urdu_cleaned = urdu_text.strip()
        
        return roman_cleaned, urdu_cleaned
    
    def extract_features(self, text: str) -> Dict:
        """Extract features from text for ML models"""
        words = self.tokenize(text)
        
        features = {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'has_numbers': bool(re.search(r'\d', text)),
            'has_special_chars': bool(re.search(r'[^a-zA-Z\s]', text)),
            'words': words,
            'unique_chars': list(set(text.lower())),
            'vowel_ratio': self._get_vowel_ratio(text),
            'consonant_ratio': self._get_consonant_ratio(text),
        }
        
        return features
    
    def _get_vowel_ratio(self, text: str) -> float:
        """Calculate ratio of vowels in text"""
        vowels = 'aeiou'
        vowel_count = sum(1 for char in text.lower() if char in vowels)
        total_chars = len([char for char in text.lower() if char.isalpha()])
        return vowel_count / total_chars if total_chars > 0 else 0
    
    def _get_consonant_ratio(self, text: str) -> float:
        """Calculate ratio of consonants in text"""
        vowels = 'aeiou'
        consonant_count = sum(1 for char in text.lower() if char.isalpha() and char not in vowels)
        total_chars = len([char for char in text.lower() if char.isalpha()])
        return consonant_count / total_chars if total_chars > 0 else 0

# Additional utility functions
def load_dictionary(file_path: str) -> Dict:
    """Load dictionary from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Dictionary file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing dictionary file: {file_path}")
        return {}

def save_dictionary(dictionary: Dict, file_path: str):
    """Save dictionary to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        print(f"Dictionary saved to: {file_path}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")

def create_reverse_dictionary(dictionary: Dict) -> Dict:
    """Create reverse mapping from Urdu to Roman"""
    reverse_dict = {}
    for roman, urdu in dictionary.items():
        if urdu not in reverse_dict:
            reverse_dict[urdu] = []
        reverse_dict[urdu].append(roman)
    return reverse_dict

def expand_dictionary_with_variations(dictionary: Dict, preprocessor: RomanUrduPreprocessor) -> Dict:
    """Expand dictionary with spelling variations"""
    expanded_dict = dictionary.copy()
    
    for word, translation in dictionary.items():
        variations = preprocessor.generate_variations(word)
        for variation in variations:
            if variation not in expanded_dict:
                expanded_dict[variation] = translation
    
    return expanded_dict

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = RomanUrduPreprocessor()
    
    test_text = "aap kesy hain? main acha hun."
    
    print("Original:", test_text)
    print("Cleaned:", preprocessor.clean_text(test_text))
    print("Normalized:", preprocessor.normalize_spelling(test_text))
    print("Tokenized:", preprocessor.tokenize(test_text))
    print("Features:", preprocessor.extract_features(test_text))
