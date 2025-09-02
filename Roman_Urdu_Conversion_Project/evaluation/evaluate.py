"""
Main evaluation script for Roman Urdu to Urdu conversion models
"""
import sys
import os
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dictionary_model import DictionaryModel
from models.ml_model import MLModel
from utils.data_loader import DataLoader
from evaluation.metrics import EvaluationMetrics, QualityAnalyzer
from evaluation.visualization import save_all_plots

class ModelEvaluator:
    """Main class for evaluating different models"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_loader = DataLoader(data_dir)
        self.results = {}
        
    def load_test_data(self) -> List[Tuple[str, str]]:
        """Load test data for evaluation"""
        test_data = self.data_loader.load_test_data()
        if not test_data:
            print("No test data found, using sample data split")
            # Use a portion of sample data as test
            sample_data = self.data_loader.load_sample_data()
            if sample_data:
                # Use last 20% as test data
                split_idx = int(0.8 * len(sample_data))
                test_data = sample_data[split_idx:]
        
        # Convert to tuple format
        test_pairs = [(item['roman'], item['urdu']) for item in test_data]
        return test_pairs
    
    def evaluate_dictionary_model(self, model_path: str = None) -> Dict:
        """Evaluate dictionary-based model"""
        print("Evaluating Dictionary Model...")
        
        # Initialize model
        model = DictionaryModel(model_path)
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            print("No test data available!")
            return {}
        
        # Generate predictions
        predictions = []
        references = []
        
        for roman_text, urdu_text in test_data:
            predicted = model.convert(roman_text)
            predictions.append(predicted)
            references.append(urdu_text)
        
        # Calculate metrics
        metrics = EvaluationMetrics.comprehensive_evaluation(predictions, references)
        
        # Additional model-specific analysis
        model_stats = model.get_statistics()
        coverage = model.get_coverage([item[0] for item in test_data])
        
        # Error analysis
        error_analysis = QualityAnalyzer.analyze_errors(predictions, references)
        length_analysis = QualityAnalyzer.length_analysis(predictions, references)
        coverage_analysis = QualityAnalyzer.coverage_analysis(
            predictions, references, model.dictionary
        )
        
        results = {
            'model_type': 'Dictionary-based',
            'metrics': metrics,
            'model_statistics': model_stats,
            'coverage': coverage,
            'error_analysis': error_analysis,
            'length_analysis': length_analysis,
            'coverage_analysis': coverage_analysis,
            'predictions': predictions[:10],  # Store first 10 for inspection
            'references': references[:10],
            'test_samples': len(test_data)
        }
        
        self.results['dictionary_model'] = results
        return results
    
    def evaluate_ml_model(self, model_type: str = "word_based") -> Dict:
        """Evaluate ML-based model"""
        print(f"Evaluating ML Model ({model_type})...")
        
        # Initialize and train model
        model = MLModel(model_type)
        model.train("sample")  # Train on sample data
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            print("No test data available!")
            return {}
        
        # Generate predictions
        predictions = []
        references = []
        
        for roman_text, urdu_text in test_data:
            predicted = model.convert(roman_text)
            predictions.append(predicted)
            references.append(urdu_text)
        
        # Calculate metrics
        metrics = EvaluationMetrics.comprehensive_evaluation(predictions, references)
        
        # Additional analysis
        error_analysis = QualityAnalyzer.analyze_errors(predictions, references)
        length_analysis = QualityAnalyzer.length_analysis(predictions, references)
        coverage_analysis = QualityAnalyzer.coverage_analysis(predictions, references)
        
        results = {
            'model_type': f'ML-based ({model_type})',
            'metrics': metrics,
            'error_analysis': error_analysis,
            'length_analysis': length_analysis,
            'coverage_analysis': coverage_analysis,
            'predictions': predictions[:10],  # Store first 10 for inspection
            'references': references[:10],
            'test_samples': len(test_data)
        }
        
        self.results[f'ml_model_{model_type}'] = results
        return results
    
    def compare_models(self) -> Dict:
        """Compare all evaluated models"""
        if not self.results:
            print("No models evaluated yet!")
            return {}
        
        comparison = {
            'model_comparison': {},
            'best_models': {},
            'metric_summary': {}
        }
        
        # Extract metrics for comparison
        for model_name, results in self.results.items():
            metrics = results.get('metrics', {})
            comparison['model_comparison'][model_name] = metrics
        
        # Find best model for each metric
        metric_names = ['word_accuracy', 'sentence_accuracy', 'bleu_score', 'rouge_l']
        
        for metric in metric_names:
            best_score = -1
            best_model = None
            
            for model_name, results in self.results.items():
                score = results.get('metrics', {}).get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            comparison['best_models'][metric] = {
                'model': best_model,
                'score': best_score
            }
        
        # Create summary table
        summary_data = []
        for model_name, results in self.results.items():
            metrics = results.get('metrics', {})
            summary_data.append({
                'Model': results.get('model_type', model_name),
                'Word Accuracy': metrics.get('word_accuracy', 0),
                'Sentence Accuracy': metrics.get('sentence_accuracy', 0),
                'BLEU Score': metrics.get('bleu_score', 0),
                'ROUGE-L': metrics.get('rouge_l', 0),
                'Character Accuracy': metrics.get('character_accuracy', 0),
                'Edit Distance': metrics.get('avg_edit_distance', 0)
            })
        
        comparison['metric_summary'] = pd.DataFrame(summary_data)
        
        return comparison
    
    def generate_report(self, output_file: str = "evaluation_report.json", images_dir: str = "report_images", md_report: str = "REPORT.md"):
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No evaluation results to report!")
            return
        
        # Add comparison results
        comparison = self.compare_models()
        
        # Create final report
        report = {
            'evaluation_summary': {
                'total_models_evaluated': len(self.results),
                'models': list(self.results.keys()),
                'evaluation_date': pd.Timestamp.now().isoformat()
            },
            'individual_results': self.results,
            'model_comparison': comparison,
            'recommendations': self.generate_recommendations()
        }
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Evaluation report saved to: {output_path}")
        
        # Also create a summary CSV
        if 'metric_summary' in comparison:
            csv_path = output_path.with_suffix('.csv')
            comparison['metric_summary'].to_csv(csv_path, index=False)
            print(f"Summary table saved to: {csv_path}")

        # Save metric plots and analysis visuals
        images_dir_path = Path(images_dir)
        images_dir_path.mkdir(parents=True, exist_ok=True)
        saved_images = save_all_plots(self.results, comparison, images_dir_path)
        if saved_images:
            print(f"Saved {len(saved_images)} images to {images_dir_path}")

        # Create a Markdown report referencing images
        try:
            md_path = Path(md_report)
            self._generate_markdown_report(md_path, report, saved_images)
            print(f"Markdown report saved to: {md_path}")
        except Exception as e:
            print(f"Failed to write Markdown report: {e}")
        
        return report

    def _generate_markdown_report(self, md_path: Path, report: Dict, saved_images: List[Path]):
        """Generate a detailed Markdown report for the project evaluation."""
        summary = report.get('evaluation_summary', {})
        comparison = report.get('model_comparison', {})
        recs = report.get('recommendations', {})

        lines = []
        lines.append("# Roman Urdu to Urdu Conversion — Evaluation Report")
        lines.append("")
        lines.append(f"Date: {summary.get('evaluation_date', '')}")
        lines.append("")
        lines.append("## Overview")
        lines.append("This report summarizes the performance of implemented models for Roman Urdu to Urdu conversion, including dictionary-based and ML-based approaches.")
        lines.append("")

        # Models evaluated
        lines.append("## Models Evaluated")
        for model_key, res in report.get('individual_results', {}).items():
            lines.append(f"- {res.get('model_type', model_key)} — Test samples: {res.get('test_samples', 0)}")
        lines.append("")

        # Metrics summary table (as text)
        if isinstance(comparison.get('metric_summary'), pd.DataFrame):
            df = comparison['metric_summary'].copy()
            lines.append("## Metric Summary")
            lines.append(df.to_markdown(index=False))
            lines.append("")

        # Insert images
        if saved_images:
            lines.append("## Visualizations")
            for p in saved_images:
                rel = Path(p).as_posix()
                lines.append(f"![{p.stem}]({rel})")
                lines.append("")

        # Recommendations
        if recs:
            lines.append("## Recommendations")
            best = recs.get('best_overall_model', {})
            if best:
                lines.append(f"- Best overall model: `{best.get('model', '')}` with weighted score {best.get('weighted_score', 0):.3f}")
            for s in recs.get('improvement_suggestions', []):
                lines.append(f"- {s}")

        md_path.write_text("\n".join(lines), encoding='utf-8')
    
    def generate_recommendations(self) -> Dict:
        """Generate recommendations based on evaluation results"""
        if not self.results:
            return {}
        
        recommendations = {
            'best_overall_model': None,
            'accuracy_leader': None,
            'speed_vs_accuracy': {},
            'improvement_suggestions': []
        }
        
        # Find best overall model (weighted average of key metrics)
        best_score = -1
        best_model = None
        
        for model_name, results in self.results.items():
            metrics = results.get('metrics', {})
            
            # Weighted score (word accuracy 40%, BLEU 30%, sentence accuracy 30%)
            score = (metrics.get('word_accuracy', 0) * 0.4 + 
                    metrics.get('bleu_score', 0) * 0.3 + 
                    metrics.get('sentence_accuracy', 0) * 0.3)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        recommendations['best_overall_model'] = {
            'model': best_model,
            'weighted_score': best_score
        }
        
        # Accuracy leader
        best_word_acc = max(
            (results.get('metrics', {}).get('word_accuracy', 0), model_name)
            for model_name, results in self.results.items()
        )
        recommendations['accuracy_leader'] = {
            'model': best_word_acc[1],
            'word_accuracy': best_word_acc[0]
        }
        
        # Generate improvement suggestions
        suggestions = []
        
        for model_name, results in self.results.items():
            metrics = results.get('metrics', {})
            word_acc = metrics.get('word_accuracy', 0)
            
            if word_acc < 0.7:
                suggestions.append(f"{model_name}: Word accuracy is low ({word_acc:.2%}). Consider expanding dictionary or improving training data.")
            
            bleu = metrics.get('bleu_score', 0)
            if bleu < 0.5:
                suggestions.append(f"{model_name}: BLEU score is low ({bleu:.3f}). Consider improving sequence-level modeling.")
            
            if 'dictionary_model' in model_name:
                coverage = results.get('coverage', {})
                if coverage.get('coverage_percentage', 0) < 80:
                    suggestions.append(f"{model_name}: Dictionary coverage is low. Add more word mappings.")
        
        recommendations['improvement_suggestions'] = suggestions
        
        return recommendations
    
    def quick_test(self, test_sentences: List[str] = None):
        """Quick test on a few sentences to see model outputs"""
        if test_sentences is None:
            test_sentences = [
                "aap kaise hain",
                "main theek hun",
                "yeh kitab achi hai",
                "ap kya kar rahe hain",
                "mujhe bhook lagi hai"
            ]
        
        print("Quick Model Test")
        print("=" * 50)
        
        # Test dictionary model
        try:
            dict_model = DictionaryModel()
            print("\nDictionary Model:")
            for sentence in test_sentences:
                converted = dict_model.convert(sentence)
                print(f"  {sentence} -> {converted}")
        except Exception as e:
            print(f"Dictionary model error: {e}")
        
        # Test ML model
        try:
            ml_model = MLModel("word_based")
            ml_model.train("sample")
            print("\nML Model (Word-based):")
            for sentence in test_sentences:
                converted = ml_model.convert(sentence)
                print(f"  {sentence} -> {converted}")
        except Exception as e:
            print(f"ML model error: {e}")

def run_full_evaluation():
    """Run complete evaluation of all models"""
    print("Starting Full Model Evaluation...")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    
    # Evaluate dictionary model
    try:
        dict_results = evaluator.evaluate_dictionary_model()
        print(f"Dictionary Model - Word Accuracy: {dict_results['metrics']['word_accuracy']:.3f}")
    except Exception as e:
        print(f"Dictionary model evaluation failed: {e}")
    
    # Evaluate ML models
    try:
        ml_word_results = evaluator.evaluate_ml_model("word_based")
        print(f"ML Word Model - Word Accuracy: {ml_word_results['metrics']['word_accuracy']:.3f}")
    except Exception as e:
        print(f"ML word model evaluation failed: {e}")
    
    try:
        ml_char_results = evaluator.evaluate_ml_model("character_based")
        print(f"ML Char Model - Word Accuracy: {ml_char_results['metrics']['word_accuracy']:.3f}")
    except Exception as e:
        print(f"ML character model evaluation failed: {e}")
    
    # Generate comprehensive report with plots and markdown
    report = evaluator.generate_report(
        output_file=str(Path("models/saved") / "evaluation_report.json"),
        images_dir=str(Path("report_images")),
        md_report=str(Path("REPORT.md"))
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if 'model_comparison' in report and 'metric_summary' in report['model_comparison']:
        print(report['model_comparison']['metric_summary'].to_string(index=False))
    
    print("\nBest Models by Metric:")
    if 'best_models' in report['model_comparison']:
        for metric, info in report['model_comparison']['best_models'].items():
            print(f"  {metric}: {info['model']} ({info['score']:.3f})")
    
    return report

if __name__ == "__main__":
    # Quick test first
    evaluator = ModelEvaluator()
    evaluator.quick_test()
    
    print("\n" + "=" * 60)
    print("Do you want to run full evaluation? (y/n): ", end="")
    
    # For automated testing, run full evaluation
    choice = "y"  # input().strip().lower()
    
    if choice == 'y':
        run_full_evaluation()
    else:
        print("Evaluation skipped.")
