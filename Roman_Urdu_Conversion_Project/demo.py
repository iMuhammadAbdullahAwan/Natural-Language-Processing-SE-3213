#!/usr/bin/env python3
"""
Demo script for Roman Urdu to Urdu Script Conversion
This script demonstrates how to use the trained models programmatically
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.dictionary_model import DictionaryModel
from models.ml_model import MLModel

def load_saved_models():
    """Load all saved models"""
    models = {}
    models_dir = Path("models/saved")
    
    print("Loading saved models...")
    
    # Load Dictionary Model
    try:
        models['dictionary'] = DictionaryModel("data/roman_urdu_dictionary.json")
        print("✓ Dictionary model loaded")
    except Exception as e:
        print(f"✗ Dictionary model failed: {e}")
    
    # Load Word-based ML Model
    try:
        word_ml = MLModel(model_type="word_based")
        word_ml.load_model(str(models_dir / "word_ml_model.pkl"))
        models['word_ml'] = word_ml
        print("✓ Word-based ML model loaded")
    except Exception as e:
        print(f"✗ Word-based ML model failed: {e}")
    
    # Load Character-based ML Model
    try:
        char_ml = MLModel(model_type="character_based")
        char_ml.load_model(str(models_dir / "char_ml_model.pkl"))
        models['char_ml'] = char_ml
        print("✓ Character-based ML model loaded")
    except Exception as e:
        print(f"✗ Character-based ML model failed: {e}")
    
    return models

def demo_conversions(models):
    """Demonstrate conversions with different models"""
    
    # Test sentences
    test_sentences = [
        "main acha hun",
        "aap kaise hain",
        "wo ghar ja raha hai",
        "hum school ja rahe hain",
        "aaj mausam bahut acha hai"
    ]
    
    print("\n" + "="*60)
    print("ROMAN URDU TO URDU CONVERSION DEMO")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Roman: {sentence}")
        print("-" * 40)
        
        for model_name, model in models.items():
            try:
                converted = model.convert_text(sentence)
                print(f"{model_name:15}: {converted}")
            except Exception as e:
                print(f"{model_name:15}: Error - {e}")
        
        print()

def interactive_demo(models):
    """Interactive conversion demo"""
    print("\n" + "="*60)
    print("INTERACTIVE CONVERSION")
    print("="*60)
    print("Enter Roman Urdu text (or 'quit' to exit)")
    
    while True:
        user_input = input("\nRoman Urdu: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        print("\nConversions:")
        print("-" * 30)
        
        for model_name, model in models.items():
            try:
                converted = model.convert_text(user_input)
                print(f"{model_name:15}: {converted}")
            except Exception as e:
                print(f"{model_name:15}: Error - {e}")

def show_model_info():
    """Show information about available models"""
    models_dir = Path("models/saved")
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    try:
        with open(models_dir / "model_registry.json", 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        for model_info in registry['available_models']:
            model_id = model_info['id']
            
            try:
                with open(models_dir / model_info['metadata_file'], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                print(f"\n{metadata['model_name']}:")
                print(f"  Type: {metadata.get('model_type', 'Unknown')}")
                print(f"  Description: {metadata.get('description', 'No description')}")
                print(f"  Status: {model_info['status']}")
                
            except Exception as e:
                print(f"\n{model_id}: Error loading metadata - {e}")
    
    except Exception as e:
        print(f"Error loading model registry: {e}")

def show_performance_metrics():
    """Show model performance metrics"""
    models_dir = Path("models/saved")
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    try:
        with open(models_dir / "evaluation_results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"{'Model':<20} {'BLEU':<8} {'ROUGE-L':<10} {'Word Acc':<10} {'Char Acc':<10}")
        print("-" * 65)
        
        for model_key, metrics in results.items():
            if model_key != 'test_info' and isinstance(metrics, dict):
                model_name = model_key.replace('_model', '').replace('_', ' ').title()
                bleu = metrics.get('BLEU', 0)
                rouge = metrics.get('ROUGE-L', 0)
                word_acc = metrics.get('Word_Accuracy', 0)
                char_acc = metrics.get('Character_Accuracy', 0)
                
                print(f"{model_name:<20} {bleu:<8.3f} {rouge:<10.3f} {word_acc:<10.3f} {char_acc:<10.3f}")
        
        print(f"\nTest set size: {results.get('test_info', {}).get('test_set_size', 'Unknown')}")
        
    except Exception as e:
        print(f"Error loading evaluation results: {e}")

def main():
    """Main demo function"""
    print("Roman Urdu to Urdu Script Conversion - Demo")
    print("=" * 50)
    
    # Show model information
    show_model_info()
    
    # Show performance metrics
    show_performance_metrics()
    
    # Load models
    models = load_saved_models()
    
    if not models:
        print("\nNo models could be loaded. Please run 'python save_models.py' first.")
        return
    
    # Demo conversions
    demo_conversions(models)
    
    # Interactive demo
    try:
        interactive_demo(models)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    print("\nDemo completed. Thank you!")
    print("\nTo use the web interface, run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
