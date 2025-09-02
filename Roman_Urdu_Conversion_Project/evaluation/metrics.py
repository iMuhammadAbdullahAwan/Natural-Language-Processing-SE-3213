"""
Evaluation metrics for Roman Urdu to Urdu conversion
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
import math

class EvaluationMetrics:
    """Class containing various evaluation metrics for transliteration"""
    
    @staticmethod
    def word_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate word-level accuracy"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_words = 0
        correct_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            
            # Align words (simple alignment)
            max_words = max(len(pred_words), len(ref_words))
            total_words += max_words
            
            for i in range(min(len(pred_words), len(ref_words))):
                if pred_words[i] == ref_words[i]:
                    correct_words += 1
        
        return correct_words / total_words if total_words > 0 else 0.0
    
    @staticmethod
    def sentence_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate sentence-level accuracy"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        correct_sentences = sum(1 for pred, ref in zip(predictions, references) 
                               if pred.strip() == ref.strip())
        
        return correct_sentences / len(predictions) if predictions else 0.0
    
    @staticmethod
    def character_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate character-level accuracy"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            # Align characters
            max_length = max(len(pred), len(ref))
            total_chars += max_length
            
            for i in range(min(len(pred), len(ref))):
                if pred[i] == ref[i]:
                    correct_chars += 1
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    @staticmethod
    def edit_distance(str1: str, str2: str) -> int:
        """Calculate edit distance (Levenshtein distance) between two strings"""
        if len(str1) < len(str2):
            return EvaluationMetrics.edit_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], max_n: int = 4) -> float:
        """Calculate BLEU score for the predictions"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_precision = []
        
        # Calculate n-gram precisions
        for n in range(1, max_n + 1):
            pred_ngrams = []
            ref_ngrams = []
            
            for pred, ref in zip(predictions, references):
                pred_words = pred.split()
                ref_words = ref.split()
                
                # Generate n-grams
                pred_ngram = [' '.join(pred_words[i:i+n]) 
                             for i in range(len(pred_words) - n + 1)]
                ref_ngram = [' '.join(ref_words[i:i+n]) 
                            for i in range(len(ref_words) - n + 1)]
                
                pred_ngrams.extend(pred_ngram)
                ref_ngrams.extend(ref_ngram)
            
            if not pred_ngrams:
                total_precision.append(0.0)
                continue
            
            # Calculate precision for this n-gram level
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            overlap = 0
            for ngram, count in pred_counter.items():
                overlap += min(count, ref_counter.get(ngram, 0))
            
            precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
            total_precision.append(precision)
        
        # Calculate geometric mean
        if any(p == 0 for p in total_precision):
            return 0.0
        
        geometric_mean = math.exp(sum(math.log(p) for p in total_precision) / len(total_precision))
        
        # Brevity penalty
        total_pred_length = sum(len(pred.split()) for pred in predictions)
        total_ref_length = sum(len(ref.split()) for ref in references)
        
        if total_pred_length == 0:
            return 0.0
        
        brevity_penalty = min(1.0, math.exp(1 - total_ref_length / total_pred_length))
        
        return geometric_mean * brevity_penalty
    
    @staticmethod
    def rouge_l(predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE-L score"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_f1 = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            
            # Find longest common subsequence
            lcs_length = EvaluationMetrics._lcs_length(pred_words, ref_words)
            
            if len(pred_words) == 0 and len(ref_words) == 0:
                f1 = 1.0
            elif len(pred_words) == 0 or len(ref_words) == 0:
                f1 = 0.0
            else:
                precision = lcs_length / len(pred_words)
                recall = lcs_length / len(ref_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_f1 += f1
        
        return total_f1 / len(predictions) if predictions else 0.0
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def meteor_score(predictions: List[str], references: List[str]) -> float:
        """Simplified METEOR score calculation"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_score = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            
            if not pred_words and not ref_words:
                score = 1.0
            elif not pred_words or not ref_words:
                score = 0.0
            else:
                matches = len(pred_words.intersection(ref_words))
                precision = matches / len(pred_words) if pred_words else 0
                recall = matches / len(ref_words) if ref_words else 0
                
                if precision + recall == 0:
                    score = 0.0
                else:
                    f_mean = (precision * recall) / (0.5 * precision + 0.5 * recall)
                    score = f_mean
            
            total_score += score
        
        return total_score / len(predictions) if predictions else 0.0
    
    @staticmethod
    def comprehensive_evaluation(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Run comprehensive evaluation with multiple metrics"""
        metrics = {
            'word_accuracy': EvaluationMetrics.word_accuracy(predictions, references),
            'sentence_accuracy': EvaluationMetrics.sentence_accuracy(predictions, references),
            'character_accuracy': EvaluationMetrics.character_accuracy(predictions, references),
            'bleu_score': EvaluationMetrics.bleu_score(predictions, references),
            'rouge_l': EvaluationMetrics.rouge_l(predictions, references),
            'meteor_score': EvaluationMetrics.meteor_score(predictions, references),
        }
        
        # Calculate average edit distance
        edit_distances = [EvaluationMetrics.edit_distance(pred, ref) 
                         for pred, ref in zip(predictions, references)]
        metrics['avg_edit_distance'] = np.mean(edit_distances) if edit_distances else 0.0
        
        return metrics

class QualityAnalyzer:
    """Analyze quality of translations at different levels"""
    
    @staticmethod
    def analyze_errors(predictions: List[str], references: List[str]) -> Dict:
        """Analyze different types of errors"""
        error_analysis = {
            'substitution_errors': 0,
            'insertion_errors': 0,
            'deletion_errors': 0,
            'word_order_errors': 0,
            'unknown_word_errors': 0,
            'total_errors': 0,
            'error_examples': []
        }
        
        for pred, ref in zip(predictions, references):
            if pred != ref:
                error_analysis['total_errors'] += 1
                
                # Simple error classification
                pred_words = pred.split()
                ref_words = ref.split()
                
                if len(pred_words) > len(ref_words):
                    error_analysis['insertion_errors'] += 1
                elif len(pred_words) < len(ref_words):
                    error_analysis['deletion_errors'] += 1
                else:
                    # Check for substitutions
                    substitutions = sum(1 for p, r in zip(pred_words, ref_words) if p != r)
                    error_analysis['substitution_errors'] += substitutions
                
                # Store error example
                if len(error_analysis['error_examples']) < 10:
                    error_analysis['error_examples'].append({
                        'prediction': pred,
                        'reference': ref,
                        'error_type': 'substitution' if len(pred_words) == len(ref_words) else 'length_mismatch'
                    })
        
        return error_analysis
    
    @staticmethod
    def coverage_analysis(predictions: List[str], references: List[str], 
                         dictionary: Dict[str, str] = None) -> Dict:
        """Analyze coverage of dictionary and unknown words"""
        all_pred_words = set()
        all_ref_words = set()
        unknown_words = set()
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            
            all_pred_words.update(pred_words)
            all_ref_words.update(ref_words)
            
            if dictionary:
                for word in pred_words:
                    if word.lower() not in dictionary:
                        unknown_words.add(word)
        
        coverage = {
            'unique_predicted_words': len(all_pred_words),
            'unique_reference_words': len(all_ref_words),
            'word_overlap': len(all_pred_words.intersection(all_ref_words)),
            'coverage_ratio': len(all_pred_words.intersection(all_ref_words)) / len(all_ref_words) if all_ref_words else 0,
            'unknown_words': list(unknown_words),
            'unknown_word_count': len(unknown_words)
        }
        
        return coverage
    
    @staticmethod
    def length_analysis(predictions: List[str], references: List[str]) -> Dict:
        """Analyze length statistics"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        analysis = {
            'avg_pred_length': np.mean(pred_lengths) if pred_lengths else 0,
            'avg_ref_length': np.mean(ref_lengths) if ref_lengths else 0,
            'length_difference': np.mean([abs(p - r) for p, r in zip(pred_lengths, ref_lengths)]),
            'length_ratio': np.mean([p / r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]),
            'min_pred_length': min(pred_lengths) if pred_lengths else 0,
            'max_pred_length': max(pred_lengths) if pred_lengths else 0,
            'min_ref_length': min(ref_lengths) if ref_lengths else 0,
            'max_ref_length': max(ref_lengths) if ref_lengths else 0,
        }
        
        return analysis

if __name__ == "__main__":
    # Test evaluation metrics
    predictions = [
        "آپ کیسے ہیں",
        "میں ٹھیک ہوں",
        "یہ کتاب اچھی ہے"
    ]
    
    references = [
        "آپ کیسے ہیں",
        "میں اچھا ہوں",
        "یہ کتاب بہت اچھی ہے"
    ]
    
    # Test all metrics
    metrics = EvaluationMetrics.comprehensive_evaluation(predictions, references)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nError Analysis:")
    errors = QualityAnalyzer.analyze_errors(predictions, references)
    for key, value in errors.items():
        if key != 'error_examples':
            print(f"{key}: {value}")
    
    print("\nLength Analysis:")
    length_stats = QualityAnalyzer.length_analysis(predictions, references)
    for key, value in length_stats.items():
        print(f"{key}: {value:.2f}")
