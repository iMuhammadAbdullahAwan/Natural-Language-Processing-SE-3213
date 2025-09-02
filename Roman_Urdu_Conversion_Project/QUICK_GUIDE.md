# Roman Urdu to Urdu Conversion Project - Quick Guide

This project provides a complete Roman Urdu to Urdu conversion system with multiple models, comprehensive evaluation, and user interfaces.

## 📁 Project Structure

```
Roman_Urdu_Conversion_Project/
├── 📊 report_images/                    # Performance metric images for reports
│   ├── metrics_overview_bar.png         # Overall model comparison
│   ├── avg_edit_distance_bar.png        # Edit distance analysis
│   ├── error_types_by_model.png         # Error pattern analysis
│   ├── avg_lengths_by_model.png         # Length statistics
│   ├── length_ratio_by_model.png        # Length ratio analysis
│   └── metric_summary_table.png         # Comprehensive metrics table
│
├── 📋 Reports & Documentation
│   ├── PROJECT_REPORT.md                # Complete detailed project report
│   ├── EVALUATION_REPORT.md             # Generated evaluation summary
│   └── README.md                        # This quick guide
│
├── 🤖 Models & Core System
│   ├── models/                          # Model implementations
│   ├── evaluation/                      # Evaluation framework
│   ├── utils/                          # Utilities and preprocessing
│   └── data/                           # Training and test data
│
└── 🖥️ User Interfaces
    ├── streamlit_app.py                 # Web interface
    └── main.py                          # Command line interface
```

## 🚀 Quick Start

### 1. Installation
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn joblib arabic-reshaper python-bidi tabulate

# For web interface
pip install streamlit
```

### 2. Run Web Interface
```bash
streamlit run streamlit_app.py
```

### 3. Run Evaluation (Generate New Images)
```bash
python evaluation/evaluate.py
```

### 4. Command Line Usage
```bash
python main.py --text "aap kaise hain" --model dictionary
```

## 📊 Performance Results

| Model | Word Accuracy | BLEU Score | ROUGE-L | Best For |
|-------|---------------|------------|---------|----------|
| **ML Word-based** | **55.9%** | **0.171** | **0.566** | Overall performance |
| Dictionary-based | 34.2% | 0.091 | 0.369 | Fast lookup, common words |
| ML Character-based | 0.0% | 0.000 | 0.000 | Needs improvement |

## 📈 Generated Images (in report_images/)

All performance metric visualizations are automatically saved as PNG files in the `report_images/` folder:

1. **metrics_overview_bar.png** - Side-by-side comparison of all models across main metrics
2. **avg_edit_distance_bar.png** - Edit distance comparison (lower is better)
3. **error_types_by_model.png** - Breakdown of error types by model
4. **avg_lengths_by_model.png** - Average text length analysis
5. **length_ratio_by_model.png** - Prediction vs reference length ratios
6. **metric_summary_table.png** - Complete metrics table as image

## 📖 Documentation

### Complete Project Report
👉 **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive 50+ page report including:
- Executive Summary
- Literature Review
- Methodology & Implementation
- Detailed Results Analysis
- Future Work & Recommendations
- Complete Technical Documentation

### Evaluation Report
👉 **[EVALUATION_REPORT.md](EVALUATION_REPORT.md)** - Auto-generated evaluation summary with:
- Performance metrics table
- Model recommendations
- Links to all visualization images

## 🔧 System Components

### Models Implemented
- **Dictionary Model**: Rule-based lookup with fuzzy matching
- **ML Word Model**: TF-IDF + Logistic Regression (word-level)
- **ML Character Model**: Context windows + Random Forest (character-level)
- **Seq2Seq Framework**: Ready for future deep learning models

### Evaluation Metrics
- Word Accuracy, Sentence Accuracy, Character Accuracy
- BLEU Score, ROUGE-L, METEOR Score
- Edit Distance, Error Type Analysis
- Length and Coverage Analysis

### User Interfaces
- **Streamlit Web App**: Interactive conversion with model selection
- **Command Line**: Batch processing and evaluation tools

## 🎯 Key Features

✅ **Multiple Model Architectures** - Dictionary, ML, and deep learning approaches  
✅ **Comprehensive Evaluation** - 6+ metrics with visual analysis  
✅ **Performance Images** - Ready-to-use charts for presentations/reports  
✅ **User-Friendly Interface** - Web app for real-time conversion  
✅ **Extensible Design** - Easy to add new models and metrics  
✅ **Complete Documentation** - Detailed technical and user documentation  

## 📊 Usage Examples

### Web Interface
1. Open `streamlit run streamlit_app.py`
2. Enter Roman Urdu text
3. Select model
4. Get instant Urdu conversion

### Evaluation
```bash
# Run full evaluation and generate new images
python evaluation/evaluate.py

# Images will be saved to report_images/
# Reports will be updated with latest results
```

### CLI Conversion
```bash
# Single text conversion
python main.py --text "main acha hun" --model ml_word

# Batch file processing
python main.py --file input.txt --output output.txt --model dictionary
```

## 🔍 For Report Writing

All necessary components for academic/technical reports:

1. **Images**: Use any from `report_images/` folder
2. **Metrics**: Reference `EVALUATION_REPORT.md` for latest numbers
3. **Methodology**: See `PROJECT_REPORT.md` sections 4-8
4. **Results**: Tables and analysis in `PROJECT_REPORT.md` section 9
5. **Code**: All source available in organized modules

## 🚀 Next Steps

1. **Immediate Use**: Images and reports are ready for presentations
2. **Model Improvement**: Add more training data and try deep learning models
3. **Feature Enhancement**: Add context awareness and better error handling
4. **Deployment**: Scale up for production use

---

**Quick Access Links:**
- 📄 [Complete Project Report](PROJECT_REPORT.md)
- 📊 [Evaluation Results](EVALUATION_REPORT.md)
- 🖼️ [Performance Images](report_images/)
- 🌐 [Web Interface](streamlit_app.py)

*Generated: September 2, 2025*
