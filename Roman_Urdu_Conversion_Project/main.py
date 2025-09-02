"""
Main application entry point for Roman Urdu to Urdu Conversion Project
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.dictionary_model import DictionaryModel
from models.ml_model import MLModel, EnsembleModel
from models.seq2seq_model import Seq2SeqTransliterator
from evaluation.evaluate import ModelEvaluator, run_full_evaluation
from evaluation.human_evaluation import main as human_eval_main
from utils.data_loader import DataLoader

class RomanUrduConverter:
    """Main application class for Roman Urdu to Urdu conversion"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.data_loader = DataLoader()
        
    def initialize_models(self):
        """Initialize all available models"""
        print("Initializing models...")
        
        try:
            # Dictionary model
            self.models['dictionary'] = DictionaryModel()
            print("✓ Dictionary model loaded")
        except Exception as e:
            print(f"✗ Dictionary model failed: {e}")
        
        try:
            # ML models
            self.models['ml_word'] = MLModel("word_based")
            self.models['ml_char'] = MLModel("character_based")
            print("✓ ML models initialized")
        except Exception as e:
            print(f"✗ ML models failed: {e}")
        
        try:
            # Seq2Seq model
            self.models['seq2seq'] = Seq2SeqTransliterator()
            print("✓ Seq2Seq model initialized")
        except Exception as e:
            print(f"✗ Seq2Seq model failed: {e}")
        
        # Set default model
        if 'dictionary' in self.models:
            self.current_model = self.models['dictionary']
        
    def train_models(self, data_source: str = "sample"):
        """Train all ML models"""
        print(f"Training models on {data_source} data...")
        
        # Train ML models
        for model_name in ['ml_word', 'ml_char']:
            if model_name in self.models:
                try:
                    print(f"Training {model_name}...")
                    self.models[model_name].train(data_source)
                    print(f"✓ {model_name} training completed")
                except Exception as e:
                    print(f"✗ {model_name} training failed: {e}")
        
        # Train Seq2Seq model (fewer epochs for demo)
        if 'seq2seq' in self.models:
            try:
                print("Training seq2seq model...")
                self.models['seq2seq'].train(data_source, epochs=10)
                print("✓ Seq2Seq training completed")
            except Exception as e:
                print(f"✗ Seq2Seq training failed: {e}")
    
    def convert_text(self, text: str, model_name: str = None) -> str:
        """Convert Roman Urdu text to Urdu"""
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.current_model:
            model = self.current_model
        else:
            return "No model available"
        
        try:
            return model.convert(text)
        except Exception as e:
            return f"Conversion error: {e}"
    
    def compare_models(self, text: str) -> dict:
        """Compare outputs from all available models"""
        results = {'input': text}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'is_trained') and not model.is_trained:
                    results[model_name] = "Model not trained"
                else:
                    results[model_name] = model.convert(text)
            except Exception as e:
                results[model_name] = f"Error: {e}"
        
        return results
    
    def interactive_mode(self):
        """Run interactive conversion mode"""
        print("\n" + "="*60)
        print("Roman Urdu to Urdu Converter - Interactive Mode")
        print("="*60)
        print("Available models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")
        print("  0. All models (comparison)")
        print("\nCommands:")
        print("  'change': Change model")
        print("  'train': Train ML models")
        print("  'eval': Run evaluation")
        print("  'quit': Exit")
        print("-"*60)
        
        while True:
            try:
                text = input("\nEnter Roman Urdu text: ").strip()
                
                if not text:
                    continue
                
                if text.lower() == 'quit':
                    break
                elif text.lower() == 'change':
                    self.change_model_interactive()
                    continue
                elif text.lower() == 'train':
                    self.train_models()
                    continue
                elif text.lower() == 'eval':
                    run_full_evaluation()
                    continue
                
                # Convert text
                if self.current_model == "all":
                    results = self.compare_models(text)
                    print(f"\nInput: {results['input']}")
                    print("-" * 40)
                    for model_name, output in results.items():
                        if model_name != 'input':
                            print(f"{model_name:12}: {output}")
                else:
                    model_name = next((name for name, model in self.models.items() 
                                     if model == self.current_model), "unknown")
                    result = self.convert_text(text)
                    print(f"\n{model_name:12}: {result}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    def change_model_interactive(self):
        """Change current model interactively"""
        print("\nAvailable models:")
        models_list = list(self.models.keys()) + ["all"]
        for i, model_name in enumerate(models_list, 1):
            print(f"  {i}. {model_name}")
        
        try:
            choice = int(input("Select model (number): ")) - 1
            if 0 <= choice < len(models_list):
                if models_list[choice] == "all":
                    self.current_model = "all"
                    print("Selected: All models (comparison mode)")
                else:
                    self.current_model = self.models[models_list[choice]]
                    print(f"Selected: {models_list[choice]}")
            else:
                print("Invalid choice")
        except (ValueError, IndexError):
            print("Invalid input")
    
    def batch_convert(self, input_file: str, output_file: str, model_name: str = None):
        """Convert texts from file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            results = []
            for line in lines:
                line = line.strip()
                if line:
                    converted = self.convert_text(line, model_name)
                    results.append(f"{line} -> {converted}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
            
            print(f"Converted {len(results)} lines. Output saved to: {output_file}")
            
        except Exception as e:
            print(f"Batch conversion failed: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Roman Urdu to Urdu Conversion Tool")
    parser.add_argument("--mode", choices=['interactive', 'convert', 'train', 'evaluate', 'human-eval'],
                       default='interactive', help="Operation mode")
    parser.add_argument("--text", type=str, help="Text to convert (for convert mode)")
    parser.add_argument("--model", choices=['dictionary', 'ml_word', 'ml_char', 'seq2seq'],
                       default='dictionary', help="Model to use")
    parser.add_argument("--input-file", type=str, help="Input file for batch conversion")
    parser.add_argument("--output-file", type=str, help="Output file for batch conversion")
    parser.add_argument("--data-source", default='sample', help="Data source for training")
    parser.add_argument("--no-train", action='store_true', help="Skip training ML models")
    
    args = parser.parse_args()
    
    # Initialize application
    app = RomanUrduConverter()
    app.initialize_models()
    
    # Execute based on mode
    if args.mode == 'interactive':
        if not args.no_train and any(name.startswith('ml') for name in app.models.keys()):
            print("Training ML models for interactive mode...")
            app.train_models(args.data_source)
        app.interactive_mode()
    
    elif args.mode == 'convert':
        if not args.text:
            print("Error: --text is required for convert mode")
            return
        
        if args.input_file and args.output_file:
            app.batch_convert(args.input_file, args.output_file, args.model)
        else:
            result = app.convert_text(args.text, args.model)
            print(f"Input:  {args.text}")
            print(f"Output: {result}")
    
    elif args.mode == 'train':
        app.train_models(args.data_source)
    
    elif args.mode == 'evaluate':
        run_full_evaluation()
    
    elif args.mode == 'human-eval':
        human_eval_main()
    
    else:
        print(f"Unknown mode: {args.mode}")

def demo():
    """Quick demonstration of the system"""
    print("Roman Urdu to Urdu Conversion - Demo")
    print("="*40)
    
    # Initialize app
    app = RomanUrduConverter()
    app.initialize_models()
    
    # Demo sentences
    demo_sentences = [
        "aap kaise hain",
        "main theek hun",
        "yeh kitab achi hai",
        "ap kya kar rahe hain",
        "mujhe bhook lagi hai",
        "pakistan ka sher kaun hai",
        "urdu ek pyari zaban hai"
    ]
    
    print("\nDictionary Model Demo:")
    print("-" * 30)
    
    for sentence in demo_sentences:
        try:
            converted = app.convert_text(sentence, 'dictionary')
            print(f"{sentence:25} -> {converted}")
        except Exception as e:
            print(f"{sentence:25} -> Error: {e}")
    
    # Train a simple ML model for demo
    print("\nTraining ML model...")
    try:
        if 'ml_word' in app.models:
            app.models['ml_word'].train('sample')
            
            print("\nML Model Demo:")
            print("-" * 30)
            
            for sentence in demo_sentences[:3]:  # Just first 3 for ML demo
                converted = app.convert_text(sentence, 'ml_word')
                print(f"{sentence:25} -> {converted}")
    except Exception as e:
        print(f"ML model demo failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo()
        print("\nFor full functionality, use:")
        print("python main.py --mode interactive")
    else:
        main()
