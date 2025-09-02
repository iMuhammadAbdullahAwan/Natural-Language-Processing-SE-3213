"""
Data loading utilities for Roman Urdu to Urdu conversion project
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader with data directory path"""
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.dictionary = None
        
    def load_dictionary(self, filename: str = "roman_urdu_dictionary.json") -> Dict:
        """Load Roman Urdu to Urdu dictionary"""
        file_path = self.data_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.dictionary = data.get('dictionary', {})
                return self.dictionary
        except FileNotFoundError:
            print(f"Dictionary file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing dictionary file: {e}")
            return {}
    
    def load_sample_data(self, filename: str = "sample_data.json") -> List[Dict]:
        """Load sample training data"""
        file_path = self.data_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.train_data = data.get('train_data', [])
                return self.train_data
        except FileNotFoundError:
            print(f"Sample data file not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing sample data file: {e}")
            return []
    
    def load_test_data(self, filename: str = "test_data.json") -> List[Dict]:
        """Load test data"""
        file_path = self.data_dir / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_data = data.get('test_data', [])
                return self.test_data
        except FileNotFoundError:
            print(f"Test data file not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing test data file: {e}")
            return []
    
    def get_parallel_data(self, data_source: str = "sample") -> Tuple[List[str], List[str]]:
        """Get parallel Roman-Urdu data as separate lists"""
        if data_source == "sample":
            data = self.train_data or self.load_sample_data()
        elif data_source == "test":
            data = self.test_data or self.load_test_data()
        else:
            raise ValueError("data_source must be 'sample' or 'test'")
        
        roman_texts = [item['roman'] for item in data]
        urdu_texts = [item['urdu'] for item in data]
        
        return roman_texts, urdu_texts
    
    def get_dataframe(self, data_source: str = "sample") -> pd.DataFrame:
        """Get data as pandas DataFrame"""
        if data_source == "sample":
            data = self.train_data or self.load_sample_data()
        elif data_source == "test":
            data = self.test_data or self.load_test_data()
        else:
            raise ValueError("data_source must be 'sample' or 'test'")
        
        return pd.DataFrame(data)
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Split sample data into train and validation sets"""
        if not self.train_data:
            self.load_sample_data()
        
        data = self.train_data.copy()
        np.random.seed(random_state)
        np.random.shuffle(data)
        
        split_index = int(len(data) * (1 - test_size))
        train_data = data[:split_index]
        val_data = data[split_index:]
        
        return train_data, val_data
    
    def create_word_pairs(self, data_source: str = "sample") -> List[Tuple[str, str]]:
        """Create word-level pairs from sentence pairs"""
        roman_texts, urdu_texts = self.get_parallel_data(data_source)
        word_pairs = []
        
        for roman_sent, urdu_sent in zip(roman_texts, urdu_texts):
            roman_words = roman_sent.split()
            urdu_words = urdu_sent.split()
            
            # Simple alignment (assuming same word order)
            min_length = min(len(roman_words), len(urdu_words))
            for i in range(min_length):
                word_pairs.append((roman_words[i], urdu_words[i]))
        
        return word_pairs
    
    def get_vocabulary(self, data_source: str = "sample", lang: str = "roman") -> set:
        """Get vocabulary from data"""
        if lang == "roman":
            texts, _ = self.get_parallel_data(data_source)
        elif lang == "urdu":
            _, texts = self.get_parallel_data(data_source)
        else:
            raise ValueError("lang must be 'roman' or 'urdu'")
        
        vocabulary = set()
        for text in texts:
            words = text.split()
            vocabulary.update(words)
        
        return vocabulary
    
    def get_character_set(self, data_source: str = "sample", lang: str = "roman") -> set:
        """Get character set from data"""
        if lang == "roman":
            texts, _ = self.get_parallel_data(data_source)
        elif lang == "urdu":
            _, texts = self.get_parallel_data(data_source)
        else:
            raise ValueError("lang must be 'roman' or 'urdu'")
        
        char_set = set()
        for text in texts:
            char_set.update(text)
        
        return char_set
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """Save processed data to JSON file"""
        file_path = self.data_dir / filename
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Data saved to: {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_custom_data(self, filepath: str) -> List[Dict]:
        """Load custom data from any JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing file: {e}")
            return []
    
    def export_to_csv(self, data_source: str = "sample", filename: Optional[str] = None):
        """Export data to CSV format"""
        df = self.get_dataframe(data_source)
        
        if filename is None:
            filename = f"{data_source}_data.csv"
        
        file_path = self.data_dir / filename
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Data exported to: {file_path}")
    
    def get_statistics(self, data_source: str = "sample") -> Dict:
        """Get statistics about the data"""
        df = self.get_dataframe(data_source)
        
        if df.empty:
            return {}
        
        roman_vocab = self.get_vocabulary(data_source, "roman")
        urdu_vocab = self.get_vocabulary(data_source, "urdu")
        roman_chars = self.get_character_set(data_source, "roman")
        urdu_chars = self.get_character_set(data_source, "urdu")
        
        stats = {
            'total_samples': len(df),
            'roman_vocabulary_size': len(roman_vocab),
            'urdu_vocabulary_size': len(urdu_vocab),
            'roman_character_set_size': len(roman_chars),
            'urdu_character_set_size': len(urdu_chars),
            'avg_roman_length': df['roman'].str.len().mean() if 'roman' in df.columns else 0,
            'avg_urdu_length': df['urdu'].str.len().mean() if 'urdu' in df.columns else 0,
            'avg_roman_words': df['roman'].str.split().str.len().mean() if 'roman' in df.columns else 0,
            'avg_urdu_words': df['urdu'].str.split().str.len().mean() if 'urdu' in df.columns else 0,
        }
        
        return stats

class DataAugmenter:
    """Class for data augmentation techniques"""
    
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor
    
    def augment_with_variations(self, data: List[Dict]) -> List[Dict]:
        """Augment data by adding spelling variations"""
        if not self.preprocessor:
            return data
        
        augmented_data = data.copy()
        
        for item in data:
            roman_text = item['roman']
            words = roman_text.split()
            
            # Generate variations for each word
            for i, word in enumerate(words):
                variations = self.preprocessor.generate_variations(word)
                for variation in variations[:2]:  # Limit to 2 variations per word
                    if variation != word:
                        new_words = words.copy()
                        new_words[i] = variation
                        new_roman = ' '.join(new_words)
                        
                        new_item = item.copy()
                        new_item['roman'] = new_roman
                        augmented_data.append(new_item)
        
        return augmented_data
    
    def add_noise(self, data: List[Dict], noise_level: float = 0.1) -> List[Dict]:
        """Add character-level noise to the data"""
        noisy_data = []
        
        for item in data:
            roman_text = item['roman']
            
            # Add character substitution noise
            if np.random.random() < noise_level:
                chars = list(roman_text)
                if chars:
                    # Random character substitution
                    idx = np.random.randint(0, len(chars))
                    chars[idx] = np.random.choice(['a', 'e', 'i', 'o', 'u'])
                    noisy_roman = ''.join(chars)
                    
                    noisy_item = item.copy()
                    noisy_item['roman'] = noisy_roman
                    noisy_data.append(noisy_item)
        
        return data + noisy_data

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader("../data")
    
    # Load and display statistics
    dictionary = loader.load_dictionary()
    train_data = loader.load_sample_data()
    test_data = loader.load_test_data()
    
    print("Dictionary size:", len(dictionary))
    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))
    
    # Display statistics
    stats = loader.get_statistics("sample")
    print("\nTraining Data Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test data split
    train_split, val_split = loader.split_data(test_size=0.2)
    print(f"\nTrain split: {len(train_split)}, Validation split: {len(val_split)}")
