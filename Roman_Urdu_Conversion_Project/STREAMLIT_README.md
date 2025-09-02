# Roman Urdu to Urdu Script Converter - Streamlit App

## Quick Start Guide

### 1. Save Models
First, run the model saving script to prepare all trained models:

```bash
python save_models.py
```

### 2. Install Streamlit Requirements
Install the required packages for the Streamlit app:

```bash
pip install -r requirements_streamlit.txt
```

### 3. Run the Streamlit App
Launch the web interface:

```bash
streamlit run streamlit_app.py
```

## Features

### 🔄 Text Converter
- **Interactive conversion**: Type or select Roman Urdu text and convert to Urdu script
- **Model selection**: Choose from Dictionary, Word-based ML, or Character-based ML models
- **Real-time conversion**: Instant results with user-friendly interface
- **Model comparison**: Compare outputs from all available models side-by-side

### 📊 Model Performance
- **Visual metrics**: Interactive charts showing BLEU, ROUGE-L, Word Accuracy, and Character Accuracy
- **Performance comparison**: Bar charts comparing all models across different metrics
- **Detailed tables**: Comprehensive performance data in tabular format
- **Metrics explanation**: Detailed descriptions of each evaluation metric
- **Best model identification**: Automatic highlighting of top-performing models

### ℹ️ Model Information
- **Model cards**: Detailed information about each available model
- **Technical specifications**: Algorithm details, feature types, and model parameters
- **Performance summaries**: Quick overview of each model's strengths
- **Status indicators**: Real-time availability status of each model

### 🧪 Batch Processing
- **Multiple input methods**: Text area, file upload, or sample data
- **Progress tracking**: Real-time progress bar during batch conversion
- **Results export**: Download conversion results as CSV files
- **Error handling**: Graceful handling of failed conversions with status reporting

## App Structure

```
streamlit_app.py
├── StreamlitRomanUrduConverter (Main Class)
│   ├── load_models()           # Load all trained models
│   ├── convert_text()          # Convert using selected model
│   ├── text_converter_page()   # Main conversion interface
│   ├── performance_page()      # Performance visualization
│   ├── model_info_page()       # Model information display
│   └── batch_processing_page() # Batch conversion interface
```

## Model Integration

The app automatically loads and integrates:

1. **Dictionary Model**: Fast rule-based conversion
2. **Word-based ML Model**: Machine learning with word-level features
3. **Character-based ML Model**: ML with character-level features
4. **Seq2Seq Model**: Deep learning model (if available)

## Performance Metrics

The app displays comprehensive evaluation metrics:

- **BLEU Score**: Translation quality measurement
- **ROUGE-L**: Longest common subsequence recall
- **Word Accuracy**: Percentage of correctly converted words
- **Character Accuracy**: Character-level conversion accuracy
- **Sentence Accuracy**: Perfect sentence conversion rate
- **Edit Distance**: Average Levenshtein distance

## UI Features

### Responsive Design
- **Multi-column layout**: Optimized for desktop and tablet viewing
- **Sidebar navigation**: Easy switching between different app sections
- **Progress indicators**: Visual feedback for long-running operations

### Urdu Text Display
- **Right-to-left text**: Proper RTL text rendering for Urdu output
- **Custom fonts**: Enhanced Urdu text display with appropriate typography
- **Highlighting**: Visual emphasis on conversion results

### Interactive Elements
- **Model comparison**: Side-by-side comparison of all models
- **Real-time conversion**: Instant feedback as you type
- **Export functionality**: Download results for further analysis

## Troubleshooting

### Common Issues

1. **Models not loading**:
   ```bash
   python save_models.py
   ```

2. **Streamlit not starting**:
   ```bash
   pip install --upgrade streamlit
   ```

3. **Missing dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

4. **Port conflicts**:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

### Model Status

Check model availability in the app:
- ✅ Ready: Model loaded and available
- ⚠️ Warning: Model loaded with limitations
- ❌ Error: Model failed to load

## Customization

### Adding New Models

1. Train your model using the existing framework
2. Save the model in `models/saved/`
3. Add metadata file with model information
4. Update `model_registry.json`
5. Restart the Streamlit app

### Modifying UI

The app uses custom CSS for styling:
- Colors and themes in the main CSS block
- Urdu text styling with RTL support
- Responsive card layouts

### Performance Optimization

- Models are loaded once at startup
- Caching is used for repeated conversions
- Batch processing with progress tracking
- Efficient memory management for large datasets

## Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment
1. Push code to GitHub repository
2. Deploy on Streamlit Cloud, Heroku, or similar platform
3. Ensure all requirements are included
4. Set up proper environment variables

## Usage Examples

### Single Text Conversion
1. Select model from sidebar
2. Enter Roman Urdu text
3. Click "Convert to Urdu"
4. View results with Urdu script output

### Batch Processing
1. Navigate to "Batch Processing" tab
2. Enter multiple sentences or upload file
3. Click "Process Batch"
4. Download results as CSV

### Performance Analysis
1. Go to "Model Performance" tab
2. View interactive charts and metrics
3. Compare models across different metrics
4. Understand model strengths and weaknesses

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all models are properly saved
3. Ensure all dependencies are installed
4. Check console output for error messages

---

**Note**: This Streamlit app provides a complete interface for the Roman Urdu to Urdu Script Conversion project, including model comparison, performance visualization, and batch processing capabilities.
