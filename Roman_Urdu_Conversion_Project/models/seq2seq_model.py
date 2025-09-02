"""
Sequence-to-Sequence model for Roman Urdu to Urdu conversion using PyTorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import RomanUrduPreprocessor
from utils.data_loader import DataLoader as CustomDataLoader
from utils.urdu_utils import UrduTextProcessor

class Vocabulary:
    """Vocabulary class for handling character/word mappings"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        vocab_list = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        vocab_list.extend(sorted(list(chars)))
        
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab_list)
        
    def text_to_indices(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to indices"""
        indices = [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) for char in text]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.char_to_idx[self.PAD_TOKEN]] * (max_length - len(indices)))
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        chars = []
        for idx in indices:
            char = self.idx_to_char.get(idx, self.UNK_TOKEN)
            if char == self.EOS_TOKEN:
                break
            if char not in [self.PAD_TOKEN, self.SOS_TOKEN]:
                chars.append(char)
        return ''.join(chars)

class TransliterationDataset(Dataset):
    """Dataset class for sequence-to-sequence transliteration"""
    
    def __init__(self, roman_texts: List[str], urdu_texts: List[str], 
                 roman_vocab: Vocabulary, urdu_vocab: Vocabulary, max_length: int = 50):
        self.roman_texts = roman_texts
        self.urdu_texts = urdu_texts
        self.roman_vocab = roman_vocab
        self.urdu_vocab = urdu_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.roman_texts)
    
    def __getitem__(self, idx):
        roman_text = self.roman_texts[idx]
        urdu_text = self.urdu_texts[idx]
        
        # Convert to indices
        roman_indices = self.roman_vocab.text_to_indices(roman_text, self.max_length)
        
        # Add SOS and EOS tokens to target
        urdu_with_tokens = self.urdu_vocab.SOS_TOKEN + urdu_text + self.urdu_vocab.EOS_TOKEN
        urdu_indices = self.urdu_vocab.text_to_indices(urdu_with_tokens, self.max_length + 2)
        
        return {
            'roman': torch.tensor(roman_indices, dtype=torch.long),
            'urdu': torch.tensor(urdu_indices, dtype=torch.long),
            'roman_text': roman_text,
            'urdu_text': urdu_text
        }

class Encoder(nn.Module):
    """LSTM Encoder for Roman Urdu text"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell

class Decoder(nn.Module):
    """LSTM Decoder for Urdu text"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.output_projection(output)
        return output, hidden, cell

class Seq2SeqModel(nn.Module):
    """Sequence-to-Sequence model for transliteration"""
    
    def __init__(self, roman_vocab_size: int, urdu_vocab_size: int, 
                 embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = Encoder(roman_vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(urdu_vocab_size, embedding_dim, hidden_dim, num_layers)
        
        self.roman_vocab_size = roman_vocab_size
        self.urdu_vocab_size = urdu_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, roman_input, urdu_input):
        batch_size = roman_input.size(0)
        
        # Encode
        encoder_output, hidden, cell = self.encoder(roman_input)
        
        # Decode
        decoder_output, _, _ = self.decoder(urdu_input, hidden, cell)
        
        return decoder_output

class Seq2SeqTransliterator:
    """Main class for sequence-to-sequence transliteration"""
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.preprocessor = RomanUrduPreprocessor()
        self.urdu_processor = UrduTextProcessor()
        self.data_loader = CustomDataLoader()
        
        # Model parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Vocabularies
        self.roman_vocab = Vocabulary()
        self.urdu_vocab = Vocabulary()
        
        # Model
        self.model = None
        self.is_trained = False
        
        # Training parameters
        self.max_length = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        
    def prepare_data(self, data_source: str = "sample"):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Load data
        roman_texts, urdu_texts = self.data_loader.get_parallel_data(data_source)
        
        if not roman_texts or not urdu_texts:
            raise ValueError("No training data available!")
        
        # Preprocess texts
        processed_roman = []
        processed_urdu = []
        
        for roman_text, urdu_text in zip(roman_texts, urdu_texts):
            roman_processed = self.preprocessor.normalize_spelling(roman_text)
            urdu_processed = self.urdu_processor.normalize_urdu_text(urdu_text)
            
            # Filter out very long sequences
            if len(roman_processed) <= self.max_length and len(urdu_processed) <= self.max_length:
                processed_roman.append(roman_processed)
                processed_urdu.append(urdu_processed)
        
        print(f"Processed {len(processed_roman)} samples")
        
        # Build vocabularies
        self.roman_vocab.build_vocab(processed_roman)
        self.urdu_vocab.build_vocab(processed_urdu)
        
        print(f"Roman vocabulary size: {self.roman_vocab.vocab_size}")
        print(f"Urdu vocabulary size: {self.urdu_vocab.vocab_size}")
        
        return processed_roman, processed_urdu
    
    def create_data_loader(self, roman_texts: List[str], urdu_texts: List[str], 
                          batch_size: int = 32, shuffle: bool = True):
        """Create PyTorch data loader"""
        dataset = TransliterationDataset(
            roman_texts, urdu_texts, 
            self.roman_vocab, self.urdu_vocab, 
            self.max_length
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def initialize_model(self):
        """Initialize the seq2seq model"""
        self.model = Seq2SeqModel(
            self.roman_vocab.vocab_size,
            self.urdu_vocab.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, data_source: str = "sample", epochs: int = 50):
        """Train the seq2seq model"""
        print("Training Seq2Seq model...")
        
        # Prepare data
        roman_texts, urdu_texts = self.prepare_data(data_source)
        
        # Split data
        split_idx = int(0.8 * len(roman_texts))
        train_roman = roman_texts[:split_idx]
        train_urdu = urdu_texts[:split_idx]
        val_roman = roman_texts[split_idx:]
        val_urdu = urdu_texts[split_idx:]
        
        # Create data loaders
        train_loader = self.create_data_loader(train_roman, train_urdu, self.batch_size)
        val_loader = self.create_data_loader(val_roman, val_urdu, self.batch_size, shuffle=False)
        
        # Initialize model
        self.initialize_model()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.urdu_vocab.char_to_idx[self.urdu_vocab.PAD_TOKEN])
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch in train_loader:
                roman_input = batch['roman'].to(self.device)
                urdu_target = batch['urdu'].to(self.device)
                
                # Teacher forcing: use previous target token as decoder input
                urdu_input = urdu_target[:, :-1]  # All except last token
                urdu_output_target = urdu_target[:, 1:]  # All except first token
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(roman_input, urdu_input)
                
                # Calculate loss
                output_reshaped = output.reshape(-1, output.size(-1))
                target_reshaped = urdu_output_target.reshape(-1)
                
                loss = criterion(output_reshaped, target_reshaped)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    roman_input = batch['roman'].to(self.device)
                    urdu_target = batch['urdu'].to(self.device)
                    
                    urdu_input = urdu_target[:, :-1]
                    urdu_output_target = urdu_target[:, 1:]
                    
                    output = self.model(roman_input, urdu_input)
                    
                    output_reshaped = output.reshape(-1, output.size(-1))
                    target_reshaped = urdu_output_target.reshape(-1)
                    
                    loss = criterion(output_reshaped, target_reshaped)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(f"best_seq2seq_model.pth")
        
        self.is_trained = True
        print("Training completed!")
    
    def convert(self, text: str) -> str:
        """Convert Roman Urdu text to Urdu using trained model"""
        if not self.is_trained or self.model is None:
            print("Model not trained!")
            return text
        
        self.model.eval()
        
        # Preprocess input
        processed_text = self.preprocessor.normalize_spelling(text)
        
        # Convert to indices
        roman_indices = self.roman_vocab.text_to_indices(processed_text, self.max_length)
        roman_tensor = torch.tensor([roman_indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Encode
            encoder_output, hidden, cell = self.model.encoder(roman_tensor)
            
            # Decode step by step
            decoder_input = torch.tensor([[self.urdu_vocab.char_to_idx[self.urdu_vocab.SOS_TOKEN]]], 
                                       dtype=torch.long).to(self.device)
            
            generated_indices = []
            
            for _ in range(self.max_length + 2):
                decoder_output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)
                
                # Get the most likely next character
                next_char_idx = decoder_output.argmax(dim=-1)
                next_char_idx_value = next_char_idx.item()
                
                # Check for EOS token
                if next_char_idx_value == self.urdu_vocab.char_to_idx[self.urdu_vocab.EOS_TOKEN]:
                    break
                
                generated_indices.append(next_char_idx_value)
                decoder_input = next_char_idx
            
            # Convert indices back to text
            converted_text = self.urdu_vocab.indices_to_text(generated_indices)
            
        return converted_text
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'roman_vocab': {
                'char_to_idx': self.roman_vocab.char_to_idx,
                'idx_to_char': self.roman_vocab.idx_to_char,
                'vocab_size': self.roman_vocab.vocab_size
            },
            'urdu_vocab': {
                'char_to_idx': self.urdu_vocab.char_to_idx,
                'idx_to_char': self.urdu_vocab.idx_to_char,
                'vocab_size': self.urdu_vocab.vocab_size
            },
            'model_params': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'max_length': self.max_length
            }
        }
        
        torch.save(save_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            save_data = torch.load(model_path, map_location=self.device)
            
            # Restore vocabularies
            self.roman_vocab.char_to_idx = save_data['roman_vocab']['char_to_idx']
            self.roman_vocab.idx_to_char = save_data['roman_vocab']['idx_to_char']
            self.roman_vocab.vocab_size = save_data['roman_vocab']['vocab_size']
            
            self.urdu_vocab.char_to_idx = save_data['urdu_vocab']['char_to_idx']
            self.urdu_vocab.idx_to_char = save_data['urdu_vocab']['idx_to_char']
            self.urdu_vocab.vocab_size = save_data['urdu_vocab']['vocab_size']
            
            # Restore model parameters
            model_params = save_data['model_params']
            self.embedding_dim = model_params['embedding_dim']
            self.hidden_dim = model_params['hidden_dim']
            self.num_layers = model_params['num_layers']
            self.max_length = model_params['max_length']
            
            # Initialize and load model
            self.initialize_model()
            self.model.load_state_dict(save_data['model_state_dict'])
            self.is_trained = True
            
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict:
        """Evaluate the model on test data"""
        if not self.is_trained:
            print("Model not trained!")
            return {}
        
        predictions = []
        references = []
        
        for roman_text, expected_urdu in test_data:
            predicted_urdu = self.convert(roman_text)
            predictions.append(predicted_urdu)
            references.append(expected_urdu)
        
        # Calculate metrics
        correct_sequences = sum(1 for pred, ref in zip(predictions, references) 
                               if pred.strip() == ref.strip())
        sequence_accuracy = correct_sequences / len(test_data) if test_data else 0
        
        # Character-level accuracy
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            for i, (p_char, r_char) in enumerate(zip(pred, ref)):
                total_chars += 1
                if p_char == r_char:
                    correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        evaluation = {
            'sequence_accuracy': sequence_accuracy,
            'character_accuracy': char_accuracy,
            'total_sequences': len(test_data),
            'correct_sequences': correct_sequences,
            'total_characters': total_chars,
            'correct_characters': correct_chars,
            'predictions': predictions,
            'references': references
        }
        
        return evaluation

if __name__ == "__main__":
    # Test the Seq2Seq model
    print("Testing Seq2Seq Model...")
    
    model = Seq2SeqTransliterator(embedding_dim=64, hidden_dim=128, num_layers=1)
    
    try:
        # Train model with fewer epochs for testing
        model.train(data_source="sample", epochs=5)
        
        # Test conversion
        test_sentences = [
            "aap kaise hain",
            "main theek hun",
            "yeh kitab achi hai"
        ]
        
        print("\nSeq2Seq Model Results:")
        for sentence in test_sentences:
            converted = model.convert(sentence)
            print(f"Roman: {sentence}")
            print(f"Urdu:  {converted}")
            print()
            
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: Seq2Seq model requires sufficient training data and computational resources")
