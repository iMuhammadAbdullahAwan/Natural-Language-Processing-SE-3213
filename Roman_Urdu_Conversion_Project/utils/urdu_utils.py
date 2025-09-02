"""
Urdu text processing utilities
"""
import re
import unicodedata
from typing import List, Dict, Tuple
import arabic_reshaper
from bidi.algorithm import get_display

class UrduTextProcessor:
    def __init__(self):
        """Initialize Urdu text processor"""
        # Urdu character ranges
        self.urdu_chars = set()
        self.urdu_chars.update(chr(i) for i in range(0x0600, 0x06FF))  # Arabic block
        self.urdu_chars.update(chr(i) for i in range(0x0750, 0x077F))  # Arabic Supplement
        self.urdu_chars.update(chr(i) for i in range(0xFB50, 0xFDFF))  # Arabic Presentation Forms-A
        self.urdu_chars.update(chr(i) for i in range(0xFE70, 0xFEFF))  # Arabic Presentation Forms-B
        
        # Common Urdu punctuation
        self.urdu_punctuation = {'۔', '؍', '؎', '؏', '؞', '؟', '٪', '٭'}
        
        # Urdu digits
        self.urdu_digits = {'۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'}
        
    def is_urdu_text(self, text: str) -> bool:
        """Check if text contains Urdu characters"""
        if not text:
            return False
        
        urdu_char_count = sum(1 for char in text if char in self.urdu_chars)
        total_chars = len([char for char in text if not char.isspace()])
        
        if total_chars == 0:
            return False
        
        # Consider text as Urdu if more than 50% characters are Urdu
        return (urdu_char_count / total_chars) > 0.5
    
    def clean_urdu_text(self, text: str) -> str:
        """Clean Urdu text by removing unwanted characters"""
        if not text:
            return ""
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-Urdu characters except spaces and basic punctuation
        cleaned_chars = []
        for char in text:
            if (char in self.urdu_chars or 
                char.isspace() or 
                char in self.urdu_punctuation or 
                char in self.urdu_digits or
                char in '.,!?;:'):
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars).strip()
    
    def normalize_urdu_text(self, text: str) -> str:
        """Normalize Urdu text"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove diacritics (Aerab)
        text = self.remove_diacritics(text)
        
        # Normalize similar characters
        text = self.normalize_characters(text)
        
        return text.strip()
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Urdu diacritics (Aerab)"""
        # Urdu diacritics unicode ranges
        diacritics = [
            '\u064B',  # FATHATAN
            '\u064C',  # DAMMATAN
            '\u064D',  # KASRATAN
            '\u064E',  # FATHA
            '\u064F',  # DAMMA
            '\u0650',  # KASRA
            '\u0651',  # SHADDA
            '\u0652',  # SUKUN
            '\u0653',  # MADDAH ABOVE
            '\u0654',  # HAMZA ABOVE
            '\u0655',  # HAMZA BELOW
            '\u0656',  # SUBSCRIPT ALEF
            '\u0657',  # INVERTED DAMMA
            '\u0658',  # MARK NOON GHUNNA
            '\u0659',  # ZWARAKAY
            '\u065A',  # VOWEL SIGN SMALL V ABOVE
            '\u065B',  # VOWEL SIGN INVERTED SMALL V ABOVE
            '\u065C',  # VOWEL SIGN DOT BELOW
            '\u065D',  # REVERSED DAMMA
            '\u065E',  # FATHA WITH TWO DOTS
            '\u065F',  # WAVY HAMZA BELOW
            '\u0670',  # SUPERSCRIPT ALEF
        ]
        
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        
        return text
    
    def normalize_characters(self, text: str) -> str:
        """Normalize similar Urdu characters"""
        normalizations = {
            'ي': 'ی',  # Arabic YEH to Urdu YEH
            'ك': 'ک',  # Arabic KAF to Urdu KAF
            'ه': 'ہ',  # Arabic HEH to Urdu HEH
            'ة': 'ہ',  # TEH MARBUTA to HEH
            '٠': '۰',  # Arabic digit zero to Urdu
            '١': '۱',  # Arabic digit one to Urdu
            '٢': '۲',  # Arabic digit two to Urdu
            '٣': '۳',  # Arabic digit three to Urdu
            '٤': '۴',  # Arabic digit four to Urdu
            '٥': '۵',  # Arabic digit five to Urdu
            '٦': '۶',  # Arabic digit six to Urdu
            '٧': '۷',  # Arabic digit seven to Urdu
            '٨': '۸',  # Arabic digit eight to Urdu
            '٩': '۹',  # Arabic digit nine to Urdu
        }
        
        for old_char, new_char in normalizations.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def tokenize_urdu(self, text: str) -> List[str]:
        """Tokenize Urdu text into words"""
        text = self.clean_urdu_text(text)
        
        # Split by whitespace and punctuation
        words = re.findall(r'[^\s\u060C\u061B\u061F\u06D4]+', text)
        
        return [word.strip() for word in words if word.strip()]
    
    def get_word_count(self, text: str) -> int:
        """Get word count in Urdu text"""
        return len(self.tokenize_urdu(text))
    
    def get_character_count(self, text: str) -> int:
        """Get character count (excluding spaces)"""
        return len([char for char in text if not char.isspace()])
    
    def reshape_urdu_text(self, text: str) -> str:
        """Reshape Urdu text for proper display (right-to-left)"""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except:
            return text
    
    def extract_urdu_features(self, text: str) -> Dict:
        """Extract features from Urdu text"""
        words = self.tokenize_urdu(text)
        
        features = {
            'word_count': len(words),
            'char_count': self.get_character_count(text),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'has_urdu_digits': any(char in self.urdu_digits for char in text),
            'has_urdu_punctuation': any(char in self.urdu_punctuation for char in text),
            'unique_chars': len(set(text.replace(' ', ''))),
            'is_pure_urdu': self.is_urdu_text(text),
        }
        
        return features
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """Compare two Urdu texts"""
        text1_clean = self.normalize_urdu_text(text1)
        text2_clean = self.normalize_urdu_text(text2)
        
        words1 = set(self.tokenize_urdu(text1_clean))
        words2 = set(self.tokenize_urdu(text2_clean))
        
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        
        comparison = {
            'exact_match': text1_clean == text2_clean,
            'word_overlap': len(common_words) / len(total_words) if total_words else 0,
            'common_words': list(common_words),
            'length_diff': abs(len(text1_clean) - len(text2_clean)),
            'char_similarity': self._calculate_char_similarity(text1_clean, text2_clean),
        }
        
        return comparison
    
    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity between two texts"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap metric
        chars1 = set(text1)
        chars2 = set(text2)
        
        common_chars = chars1.intersection(chars2)
        total_chars = chars1.union(chars2)
        
        return len(common_chars) / len(total_chars) if total_chars else 0
    
    def convert_digits_to_urdu(self, text: str) -> str:
        """Convert English digits to Urdu digits"""
        digit_map = {
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }
        
        for eng_digit, urdu_digit in digit_map.items():
            text = text.replace(eng_digit, urdu_digit)
        
        return text
    
    def convert_digits_to_english(self, text: str) -> str:
        """Convert Urdu digits to English digits"""
        digit_map = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        for urdu_digit, eng_digit in digit_map.items():
            text = text.replace(urdu_digit, eng_digit)
        
        return text

class UrduRomanTransliterator:
    """Simple Urdu to Roman transliteration (reverse of our main task)"""
    
    def __init__(self):
        # Basic character mappings for reverse transliteration
        self.urdu_to_roman = {
            'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't',
            'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
            'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh',
            'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
            'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
            'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w', 'ہ': 'h',
            'ھ': 'h', 'ء': '', 'ی': 'y', 'ے': 'e', '۔': '.',
        }
    
    def transliterate_to_roman(self, urdu_text: str) -> str:
        """Convert Urdu text to Roman (approximate)"""
        roman_chars = []
        
        for char in urdu_text:
            if char in self.urdu_to_roman:
                roman_chars.append(self.urdu_to_roman[char])
            elif char.isspace():
                roman_chars.append(char)
            else:
                roman_chars.append(char)  # Keep unknown characters as-is
        
        return ''.join(roman_chars)

if __name__ == "__main__":
    # Test the Urdu processor
    processor = UrduTextProcessor()
    
    test_urdu = "آپ کیسے ہیں؟ میں ٹھیک ہوں۔"
    
    print("Original:", test_urdu)
    print("Is Urdu:", processor.is_urdu_text(test_urdu))
    print("Cleaned:", processor.clean_urdu_text(test_urdu))
    print("Normalized:", processor.normalize_urdu_text(test_urdu))
    print("Tokenized:", processor.tokenize_urdu(test_urdu))
    print("Features:", processor.extract_urdu_features(test_urdu))
    
    # Test transliteration
    transliterator = UrduRomanTransliterator()
    print("Roman approximation:", transliterator.transliterate_to_roman(test_urdu))
