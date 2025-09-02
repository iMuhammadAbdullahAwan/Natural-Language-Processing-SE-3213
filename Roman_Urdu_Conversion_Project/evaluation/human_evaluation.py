"""
Human evaluation interface for Roman Urdu to Urdu conversion
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dictionary_model import DictionaryModel
from models.ml_model import MLModel
from utils.data_loader import DataLoader

class HumanEvaluationGUI:
    """GUI for human evaluation of transliteration quality"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Roman Urdu to Urdu - Human Evaluation")
        self.root.geometry("800x600")
        
        # Initialize models
        self.dict_model = DictionaryModel()
        self.ml_model = None
        self.data_loader = DataLoader()
        
        # Evaluation data
        self.test_sentences = []
        self.current_index = 0
        self.evaluations = []
        
        # Load test data
        self.load_test_data()
        
        # Setup GUI
        self.setup_gui()
        
        # Configure fonts for Urdu text
        self.urdu_font = ("Arial Unicode MS", 12)
        
    def load_test_data(self):
        """Load test sentences for evaluation"""
        try:
            test_data = self.data_loader.load_test_data()
            if test_data:
                self.test_sentences = [(item['roman'], item['urdu']) for item in test_data]
            else:
                # Fallback to sample sentences
                self.test_sentences = [
                    ("aap kaise hain", "آپ کیسے ہیں"),
                    ("main theek hun", "میں ٹھیک ہوں"),
                    ("yeh kitab achi hai", "یہ کتاب اچھی ہے"),
                    ("ap kya kar rahe hain", "آپ کیا کر رہے ہیں"),
                    ("mujhe bhook lagi hai", "مجھے بھوک لگی ہے")
                ]
        except Exception as e:
            print(f"Error loading test data: {e}")
            self.test_sentences = [
                ("aap kaise hain", "آپ کیسے ہیں"),
                ("main theek hun", "میں ٹھیک ہوں")
            ]
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Human Evaluation - Roman Urdu to Urdu Conversion", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Progress
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(input_frame, text="Roman Urdu:").grid(row=0, column=0, sticky=tk.W)
        self.roman_text = tk.Text(input_frame, height=2, width=60, font=("Arial", 11))
        self.roman_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Model outputs section
        outputs_frame = ttk.LabelFrame(main_frame, text="Model Outputs", padding="10")
        outputs_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Dictionary model output
        ttk.Label(outputs_frame, text="Dictionary Model:").grid(row=0, column=0, sticky=tk.W)
        self.dict_output = tk.Text(outputs_frame, height=2, width=60, font=self.urdu_font)
        self.dict_output.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 5))
        
        # ML model output
        ttk.Label(outputs_frame, text="ML Model:").grid(row=2, column=0, sticky=tk.W)
        self.ml_output = tk.Text(outputs_frame, height=2, width=60, font=self.urdu_font)
        self.ml_output.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 5))
        
        # Reference section
        ref_frame = ttk.LabelFrame(main_frame, text="Reference Translation", padding="10")
        ref_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.reference_text = tk.Text(ref_frame, height=2, width=60, font=self.urdu_font)
        self.reference_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Evaluation section
        eval_frame = ttk.LabelFrame(main_frame, text="Evaluation", padding="10")
        eval_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Rating scales
        ttk.Label(eval_frame, text="Dictionary Model Rating:").grid(row=0, column=0, sticky=tk.W)
        self.dict_rating = ttk.Scale(eval_frame, from_=1, to=5, orient=tk.HORIZONTAL, length=200)
        self.dict_rating.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.dict_rating_label = ttk.Label(eval_frame, text="3")
        self.dict_rating_label.grid(row=0, column=2, padx=(10, 0))
        
        ttk.Label(eval_frame, text="ML Model Rating:").grid(row=1, column=0, sticky=tk.W)
        self.ml_rating = ttk.Scale(eval_frame, from_=1, to=5, orient=tk.HORIZONTAL, length=200)
        self.ml_rating.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.ml_rating_label = ttk.Label(eval_frame, text="3")
        self.ml_rating_label.grid(row=1, column=2, padx=(10, 0))
        
        # Comments
        ttk.Label(eval_frame, text="Comments:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.comments = tk.Text(eval_frame, height=3, width=60)
        self.comments.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(button_frame, text="Previous", command=self.previous_sentence).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Next", command=self.next_sentence).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Save Evaluation", command=self.save_evaluation).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="Export Results", command=self.export_results).grid(row=0, column=3)
        
        # Bind scale events
        self.dict_rating.configure(command=self.update_dict_rating)
        self.ml_rating.configure(command=self.update_ml_rating)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Load first sentence
        self.load_current_sentence()
    
    def update_dict_rating(self, value):
        """Update dictionary rating label"""
        self.dict_rating_label.config(text=f"{float(value):.1f}")
    
    def update_ml_rating(self, value):
        """Update ML rating label"""
        self.ml_rating_label.config(text=f"{float(value):.1f}")
    
    def load_current_sentence(self):
        """Load current sentence for evaluation"""
        if not self.test_sentences:
            return
        
        roman, reference = self.test_sentences[self.current_index]
        
        # Update progress
        self.progress_label.config(
            text=f"Sentence {self.current_index + 1} of {len(self.test_sentences)}"
        )
        
        # Clear and populate fields
        self.roman_text.delete(1.0, tk.END)
        self.roman_text.insert(1.0, roman)
        
        self.reference_text.delete(1.0, tk.END)
        self.reference_text.insert(1.0, reference)
        
        # Generate model outputs
        try:
            dict_output = self.dict_model.convert(roman)
            self.dict_output.delete(1.0, tk.END)
            self.dict_output.insert(1.0, dict_output)
        except Exception as e:
            self.dict_output.delete(1.0, tk.END)
            self.dict_output.insert(1.0, f"Error: {e}")
        
        # ML model output (if available)
        if self.ml_model is None:
            try:
                self.ml_model = MLModel("word_based")
                self.ml_model.train("sample")
            except:
                pass
        
        if self.ml_model:
            try:
                ml_output = self.ml_model.convert(roman)
                self.ml_output.delete(1.0, tk.END)
                self.ml_output.insert(1.0, ml_output)
            except Exception as e:
                self.ml_output.delete(1.0, tk.END)
                self.ml_output.insert(1.0, f"Error: {e}")
        else:
            self.ml_output.delete(1.0, tk.END)
            self.ml_output.insert(1.0, "ML model not available")
        
        # Reset ratings
        self.dict_rating.set(3)
        self.ml_rating.set(3)
        self.comments.delete(1.0, tk.END)
        
        # Load previous evaluation if exists
        self.load_previous_evaluation()
    
    def load_previous_evaluation(self):
        """Load previous evaluation for current sentence if exists"""
        for eval_data in self.evaluations:
            if eval_data['index'] == self.current_index:
                self.dict_rating.set(eval_data['dict_rating'])
                self.ml_rating.set(eval_data['ml_rating'])
                self.comments.delete(1.0, tk.END)
                self.comments.insert(1.0, eval_data['comments'])
                break
    
    def save_evaluation(self):
        """Save current evaluation"""
        if not self.test_sentences:
            return
        
        # Remove any existing evaluation for this index
        self.evaluations = [e for e in self.evaluations if e['index'] != self.current_index]
        
        # Add new evaluation
        roman, reference = self.test_sentences[self.current_index]
        dict_output = self.dict_output.get(1.0, tk.END).strip()
        ml_output = self.ml_output.get(1.0, tk.END).strip()
        
        evaluation = {
            'index': self.current_index,
            'roman_input': roman,
            'reference': reference,
            'dict_output': dict_output,
            'ml_output': ml_output,
            'dict_rating': float(self.dict_rating.get()),
            'ml_rating': float(self.ml_rating.get()),
            'comments': self.comments.get(1.0, tk.END).strip()
        }
        
        self.evaluations.append(evaluation)
        messagebox.showinfo("Saved", "Evaluation saved successfully!")
    
    def next_sentence(self):
        """Move to next sentence"""
        if self.current_index < len(self.test_sentences) - 1:
            self.current_index += 1
            self.load_current_sentence()
    
    def previous_sentence(self):
        """Move to previous sentence"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_sentence()
    
    def export_results(self):
        """Export evaluation results"""
        if not self.evaluations:
            messagebox.showwarning("No Data", "No evaluations to export!")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
            elif file_path.endswith('.csv'):
                df = pd.DataFrame(self.evaluations)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("Exported", f"Results exported to {file_path}")
            
            # Also generate summary
            self.generate_summary(file_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def generate_summary(self, base_path: str):
        """Generate evaluation summary"""
        if not self.evaluations:
            return
        
        summary = {
            'total_evaluations': len(self.evaluations),
            'average_dict_rating': sum(e['dict_rating'] for e in self.evaluations) / len(self.evaluations),
            'average_ml_rating': sum(e['ml_rating'] for e in self.evaluations) / len(self.evaluations),
            'dict_rating_distribution': {},
            'ml_rating_distribution': {},
            'common_issues': []
        }
        
        # Calculate rating distributions
        for rating in range(1, 6):
            dict_count = sum(1 for e in self.evaluations if abs(e['dict_rating'] - rating) < 0.5)
            ml_count = sum(1 for e in self.evaluations if abs(e['ml_rating'] - rating) < 0.5)
            summary['dict_rating_distribution'][str(rating)] = dict_count
            summary['ml_rating_distribution'][str(rating)] = ml_count
        
        # Extract common issues from comments
        comments = [e['comments'] for e in self.evaluations if e['comments']]
        summary['sample_comments'] = comments[:5]  # First 5 comments
        
        # Save summary
        summary_path = Path(base_path).with_suffix('.summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

class CommandLineEvaluation:
    """Command line interface for human evaluation"""
    
    def __init__(self):
        self.dict_model = DictionaryModel()
        self.ml_model = None
        self.data_loader = DataLoader()
        self.evaluations = []
    
    def run_evaluation(self, num_samples: int = 10):
        """Run command line evaluation"""
        print("Human Evaluation - Roman Urdu to Urdu Conversion")
        print("=" * 50)
        print("Rate each translation on a scale of 1-5:")
        print("1 = Very Poor, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent")
        print("Enter 'q' to quit at any time\n")
        
        # Load test data
        test_data = self.data_loader.load_test_data()
        if not test_data:
            test_data = [
                {"roman": "aap kaise hain", "urdu": "آپ کیسے ہیں"},
                {"roman": "main theek hun", "urdu": "میں ٹھیک ہوں"}
            ]
        
        test_data = test_data[:num_samples]
        
        # Initialize ML model
        try:
            self.ml_model = MLModel("word_based")
            self.ml_model.train("sample")
        except:
            print("Warning: ML model not available")
        
        for i, item in enumerate(test_data):
            print(f"\nSample {i+1}/{len(test_data)}")
            print("-" * 30)
            print(f"Roman Input: {item['roman']}")
            print(f"Reference:   {item['urdu']}")
            
            # Generate predictions
            dict_output = self.dict_model.convert(item['roman'])
            print(f"Dictionary:  {dict_output}")
            
            if self.ml_model:
                ml_output = self.ml_model.convert(item['roman'])
                print(f"ML Model:    {ml_output}")
            else:
                ml_output = "N/A"
            
            # Get ratings
            try:
                dict_rating = self.get_rating("Rate Dictionary model (1-5): ")
                if dict_rating is None:
                    break
                
                if self.ml_model:
                    ml_rating = self.get_rating("Rate ML model (1-5): ")
                    if ml_rating is None:
                        break
                else:
                    ml_rating = 0
                
                comments = input("Comments (optional): ").strip()
                
                # Save evaluation
                evaluation = {
                    'roman_input': item['roman'],
                    'reference': item['urdu'],
                    'dict_output': dict_output,
                    'ml_output': ml_output,
                    'dict_rating': dict_rating,
                    'ml_rating': ml_rating,
                    'comments': comments
                }
                
                self.evaluations.append(evaluation)
                
            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user")
                break
        
        # Save results
        if self.evaluations:
            self.save_results()
    
    def get_rating(self, prompt: str) -> int:
        """Get rating from user input"""
        while True:
            try:
                response = input(prompt).strip()
                if response.lower() == 'q':
                    return None
                
                rating = int(response)
                if 1 <= rating <= 5:
                    return rating
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
    
    def save_results(self):
        """Save evaluation results"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_evaluation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        # Print summary
        if self.evaluations:
            dict_avg = sum(e['dict_rating'] for e in self.evaluations) / len(self.evaluations)
            ml_ratings = [e['ml_rating'] for e in self.evaluations if e['ml_rating'] > 0]
            ml_avg = sum(ml_ratings) / len(ml_ratings) if ml_ratings else 0
            
            print(f"Summary:")
            print(f"  Total evaluations: {len(self.evaluations)}")
            print(f"  Dictionary model average rating: {dict_avg:.2f}")
            if ml_avg > 0:
                print(f"  ML model average rating: {ml_avg:.2f}")

def main():
    """Main function to choose evaluation interface"""
    print("Roman Urdu to Urdu Conversion - Human Evaluation")
    print("Choose evaluation interface:")
    print("1. GUI Interface")
    print("2. Command Line Interface")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            app = HumanEvaluationGUI()
            app.run()
        except Exception as e:
            print(f"GUI failed to start: {e}")
            print("Falling back to command line interface...")
            evaluator = CommandLineEvaluation()
            evaluator.run_evaluation()
    elif choice == "2":
        evaluator = CommandLineEvaluation()
        num_samples = input("Number of samples to evaluate (default 10): ").strip()
        try:
            num_samples = int(num_samples) if num_samples else 10
        except ValueError:
            num_samples = 10
        
        evaluator.run_evaluation(num_samples)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
