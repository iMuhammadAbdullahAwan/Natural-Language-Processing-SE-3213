import streamlit as st
import json
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.dictionary_model import DictionaryModel
from models.ml_model import MLModel
from utils.preprocessing import RomanUrduPreprocessor

# Page configuration
st.set_page_config(
    page_title="Roman Urdu to Urdu Converter",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* === STREAMLIT APP BASE STYLING === */
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* === HEADER AND NAVBAR FIXES === */
    /* Main app header */
    header[data-testid="stHeader"] {
        background-color: #2E86AB !important;
        color: white !important;
        height: 3rem !important;
    }
    
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    
    /* Streamlit menu button in header */
    .stApp > header button {
        color: white !important;
    }
    
    /* Top navigation bar */
    .stApp > header {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    /* Header toolbar */
    .stToolbar {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    .stToolbar * {
        color: white !important;
    }
    
    /* === SELECT OPTIONS AND DROPDOWN FIXES === */
    /* All selectbox containers */
    .stSelectbox {
        color: #262730 !important;
    }
    
    .stSelectbox label {
        color: #262730 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Selectbox dropdown arrow and container */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
        border-radius: 5px !important;
    }
    
    /* Selectbox text */
    .stSelectbox > div > div > div {
        color: #262730 !important;
    }
    
    /* Dropdown options */
    .stSelectbox option {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Select dropdown when opened */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    [data-baseweb="select"] * {
        color: #262730 !important;
    }
    
    /* Select options list */
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #2E86AB !important;
    }
    
    [data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
    }
    
    /* Multi-select options */
    .stMultiSelect {
        color: #262730 !important;
    }
    
    .stMultiSelect label {
        color: #262730 !important;
        font-weight: 600 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
    }
    
    /* === PAGE TITLE AND NAVIGATION === */
    /* Main page title */
    .main h1 {
        color: #2E86AB !important;
        text-align: center !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        margin-bottom: 2rem !important;
    }
    
    /* Section headers */
    .main h2, .main h3, .main h4 {
        color: #2E86AB !important;
        font-weight: 600 !important;
    }
    
    /* Tab navigation if using st.tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #262730 !important;
        background-color: #f0f2f6 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #2E86AB !important;
        background-color: #e9ecef !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2E86AB !important;
        background-color: #ffffff !important;
    }
    
    /* === INPUT FIELD IMPROVEMENTS === */
    /* Text input styling */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
        border-radius: 5px !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.25) !important;
    }
    
    .stTextInput label {
        color: #262730 !important;
        font-weight: 600 !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
        border-radius: 5px !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.25) !important;
    }
    
    .stTextArea label {
        color: #262730 !important;
        font-weight: 600 !important;
    }
    
    /* === FORCE PROPER TEXT VISIBILITY === */
    /* All text elements */
    .main .block-container {
        color: #262730 !important;
    }
    
    .main .block-container * {
        color: #262730 !important;
    }
    
    /* Streamlit native elements */
    .stMarkdown, .stText, .stWrite {
        color: #262730 !important;
    }
    
    .stMarkdown *, .stText *, .stWrite * {
        color: #262730 !important;
    }
    
    /* Headers with forced colors */
    h1, h2, h3, h4, h5, h6 {
        color: #2E86AB !important;
    }
    
    .main-header {
        text-align: center;
        color: #2E86AB !important;
        font-size: 2.5em;
        margin-bottom: 30px;
        font-weight: bold;
    }
    
    /* Model cards with high contrast */
    .model-card {
        background-color: #f8f9fa !important;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #e9ecef;
        color: #212529 !important;
    }
    
    .model-card * {
        color: #212529 !important;
    }
    
    /* Result boxes */
    .result-box {
        background-color: #e3f2fd !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
        color: #1565c0 !important;
    }
    
    .result-box * {
        color: #1565c0 !important;
    }
    
    /* Urdu text with strong contrast */
    .urdu-text {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif !important;
        font-size: 1.4em !important;
        text-align: right !important;
        direction: rtl !important;
        color: #000000 !important;
        background-color: #f8f9fa !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #007bff !important;
        font-weight: bold !important;
    }
    
    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card * {
        color: white !important;
    }
    
    /* Input elements */
    .stTextInput input, .stTextArea textarea {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }
    
    .stSelectbox select {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    /* Sidebar styling with comprehensive element coverage */
    .css-1d391kg {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    .css-1d391kg * {
        color: #262730 !important;
    }
    
    /* Sidebar specific elements */
    .sidebar .sidebar-content {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Sidebar navigation and headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #2E86AB !important;
    }
    
    /* Sidebar text elements */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div, 
    .css-1d391kg label, .css-1d391kg li {
        color: #262730 !important;
    }
    
    /* Checkboxes in sidebar */
    .css-1d391kg .stCheckbox {
        color: #262730 !important;
    }
    
    .css-1d391kg .stCheckbox label {
        color: #262730 !important;
    }
    
    .css-1d391kg .stCheckbox input[type="checkbox"] {
        accent-color: #2E86AB !important;
    }
    
    /* Radio buttons in sidebar */
    .css-1d391kg .stRadio {
        color: #262730 !important;
    }
    
    .css-1d391kg .stRadio label {
        color: #262730 !important;
    }
    
    .css-1d391kg .stRadio input[type="radio"] {
        accent-color: #2E86AB !important;
    }
    
    /* Selectboxes in sidebar */
    .css-1d391kg .stSelectbox {
        color: #262730 !important;
    }
    
    .css-1d391kg .stSelectbox label {
        color: #262730 !important;
    }
    
    .css-1d391kg .stSelectbox select {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }
    
    /* Navigation elements */
    .css-1d391kg .nav-link {
        color: #262730 !important;
    }
    
    .css-1d391kg .nav-link:hover {
        color: #2E86AB !important;
        background-color: #e9ecef !important;
    }
    
    /* Buttons in sidebar */
    .css-1d391kg .stButton > button {
        background-color: #2E86AB !important;
        color: white !important;
        border: 1px solid #2E86AB !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background-color: #1f5f7a !important;
        color: white !important;
    }
    
    /* Input elements in sidebar */
    .css-1d391kg .stTextInput input, 
    .css-1d391kg .stTextArea textarea,
    .css-1d391kg .stNumberInput input {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }
    
    /* Slider elements in sidebar */
    .css-1d391kg .stSlider {
        color: #262730 !important;
    }
    
    .css-1d391kg .stSlider label {
        color: #262730 !important;
    }
    
    /* File uploader in sidebar */
    .css-1d391kg .stFileUploader {
        color: #262730 !important;
    }
    
    .css-1d391kg .stFileUploader label {
        color: #262730 !important;
    }
    
    /* Multiselect in sidebar */
    .css-1d391kg .stMultiSelect {
        color: #262730 !important;
    }
    
    .css-1d391kg .stMultiSelect label {
        color: #262730 !important;
    }
    
    /* Any remaining sidebar elements */
    .sidebar * {
        color: #262730 !important;
    }
    
    /* Alternative sidebar class names (Streamlit versions) */
    .css-17eq0hr, .css-1lcbmhc, .css-1d391kg, 
    .stSidebar, [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    .css-17eq0hr *, .css-1lcbmhc *, .css-1d391kg *, 
    .stSidebar *, [data-testid="stSidebar"] * {
        color: #262730 !important;
    }
    
    /* Checkbox and radio specific fixes */
    input[type="checkbox"], input[type="radio"] {
        accent-color: #2E86AB !important;
        width: 16px !important;
        height: 16px !important;
    }
    
    /* Label text for all form elements */
    .stCheckbox > label, .stRadio > label, 
    .stSelectbox > label, .stTextInput > label,
    .stTextArea > label, .stSlider > label {
        color: #262730 !important;
        font-weight: 500 !important;
    }
    
    /* Tables and dataframes */
    .dataframe {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    .dataframe * {
        color: #262730 !important;
    }
    
    /* Metric widgets */
    .metric-container {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    .metric-container * {
        color: #262730 !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    .stSuccess * {
        color: #155724 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
    }
    
    .stInfo * {
        color: #0c5460 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
    }
    
    .stWarning * {
        color: #856404 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2E86AB !important;
        color: white !important;
        border: none !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1f5f7a !important;
        color: white !important;
    }
    
    /* Force dark text on any element that might be white */
    p, span, div, label, li {
        color: #262730 !important;
    }
    
    /* Exception for specific gradient backgrounds */
    .metric-card p, .metric-card span, .metric-card div {
        color: white !important;
    }
    
    /* Modern Streamlit sidebar data attributes */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #262730 !important;
    }
    
    /* Modern checkbox styling */
    [data-testid="stCheckbox"] {
        color: #262730 !important;
    }
    
    [data-testid="stCheckbox"] label {
        color: #262730 !important;
    }
    
    [data-testid="stCheckbox"] span {
        color: #262730 !important;
    }
    
    /* Modern radio button styling */
    [data-testid="stRadio"] {
        color: #262730 !important;
    }
    
    [data-testid="stRadio"] label {
        color: #262730 !important;
    }
    
    [data-testid="stRadio"] span {
        color: #262730 !important;
    }
    
    /* Modern selectbox styling */
    [data-testid="stSelectbox"] {
        color: #262730 !important;
    }
    
    [data-testid="stSelectbox"] label {
        color: #262730 !important;
    }
    
    /* Form element labels across all contexts */
    .stFormSubmitButton, .stDownloadButton, .stFileUploader,
    .stCameraInput, .stColorPicker, .stDateInput, .stTimeInput {
        color: #262730 !important;
    }
    
    .stFormSubmitButton *, .stDownloadButton *, .stFileUploader *,
    .stCameraInput *, .stColorPicker *, .stDateInput *, .stTimeInput * {
        color: #262730 !important;
    }
    
    /* Widget labels specifically */
    .Widget > label {
        color: #262730 !important;
    }
    
    /* Ensure all form text is visible */
    .stForm {
        color: #262730 !important;
    }
    
    .stForm * {
        color: #262730 !important;
    }
    
    /* === ADDITIONAL HEADER AND NAVBAR FIXES === */
    /* Streamlit header toolbar */
    div[data-testid="stToolbar"] {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    div[data-testid="stToolbar"] * {
        color: white !important;
    }
    
    /* Main navigation */
    .stApp header {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    .stApp header * {
        color: white !important;
    }
    
    /* Streamlit menu button */
    button[data-testid="stMainMenuButton"] {
        color: white !important;
    }
    
    /* Header status indicators */
    .stStatus {
        color: white !important;
    }
    
    /* === COMPREHENSIVE SELECT OPTION FIXES === */
    /* All select elements - covering multiple versions */
    select {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
        border-radius: 5px !important;
        font-size: 1rem !important;
        padding: 8px 12px !important;
    }
    
    select option {
        background-color: #ffffff !important;
        color: #262730 !important;
        padding: 8px !important;
    }
    
    select option:hover {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
    }
    
    /* Selectbox current value display */
    .stSelectbox div div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #2E86AB !important;
    }
    
    /* Selectbox dropdown button */
    .stSelectbox div div[data-baseweb="select"] svg {
        color: #2E86AB !important;
    }
    
    /* Option list container */
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 2px solid #2E86AB !important;
        border-radius: 5px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    ul[data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #262730 !important;
        padding: 10px 15px !important;
        font-size: 1rem !important;
    }
    
    ul[data-baseweb="menu"] li:hover {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
    }
    
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    /* === NAVBAR AND NAVIGATION ELEMENTS === */
    /* Tab navigation */
    .stTabs [data-baseweb="tab-list"] button {
        color: #262730 !important;
        background-color: #f0f2f6 !important;
        border: 1px solid #e9ecef !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #2E86AB !important;
        background-color: #e9ecef !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2E86AB !important;
        background-color: #ffffff !important;
        border-bottom: 3px solid #2E86AB !important;
        font-weight: 600 !important;
    }
    
    /* === FORCE ALL INTERACTIVE ELEMENTS === */
    /* Force visibility on all clickable elements */
    button, input, select, textarea, label {
        color: #262730 !important;
    }
    
    /* Exception for specific styled buttons */
    .stButton > button {
        background-color: #2E86AB !important;
        color: white !important;
    }
    
    /* Force header elements to white */
    header *, [data-testid="stHeader"] *, [data-testid="stToolbar"] * {
        color: white !important;
    }
    
    /* Override any remaining white text issues */
    .main * {
        color: #262730 !important;
    }
    
    /* Keep metric cards white text */
    .metric-card, .metric-card * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRomanUrduConverter:
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.evaluation_results = {}
        self.preprocessor = RomanUrduPreprocessor()
        self.load_models()
        
    def load_models(self):
        """Load all available models"""
        models_dir = Path("models/saved")
        
        try:
            # Load model registry
            with open(models_dir / "model_registry.json", 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
            
            # Load evaluation results
            with open(models_dir / "evaluation_results.json", 'r', encoding='utf-8') as f:
                self.evaluation_results = json.load(f)
            
            # Load each model
            for model_info in self.registry['available_models']:
                model_id = model_info['id']
                
                try:
                    # Load metadata
                    with open(models_dir / model_info['metadata_file'], 'r', encoding='utf-8') as f:
                        self.model_metadata[model_id] = json.load(f)
                    
                    # Load model based on type
                    if model_id == 'dictionary':
                        self.models[model_id] = DictionaryModel("data/roman_urdu_dictionary.json")
                    elif model_id in ['word_ml', 'char_ml']:
                        model_type = "word_based" if model_id == 'word_ml' else "character_based"
                        model = MLModel(model_type=model_type)
                        model.load_model(str(models_dir / model_info['file']))
                        self.models[model_id] = model
                    elif model_id == 'seq2seq':
                        # Skip seq2seq for now as it requires more complex loading
                        continue
                        
                except Exception as e:
                    st.warning(f"Could not load {model_id} model: {e}")
                    
        except Exception as e:
            st.error(f"Error loading models: {e}")
            # Create fallback
            self.models['dictionary'] = DictionaryModel("data/roman_urdu_dictionary.json")
            self.model_metadata['dictionary'] = {
                'model_name': 'Dictionary-Based Converter',
                'description': 'Fallback dictionary model'
            }

    def convert_text(self, text, model_id):
        """Convert text using specified model"""
        if model_id not in self.models:
            return "Model not available"
        
        try:
            if model_id == 'dictionary':
                return self.models[model_id].convert_text(text)
            else:
                # For ML models, use convert_text method
                return self.models[model_id].convert_text(text)
        except Exception as e:
            return f"Conversion error: {e}"

    def run(self):
        """Main Streamlit application"""
        # Header
        st.markdown('<h1 class="main-header">🔤 Roman Urdu to Urdu Script Converter</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Model selection
            available_models = list(self.models.keys())
            model_names = [self.model_metadata.get(m, {}).get('model_name', m) for m in available_models]
            
            selected_model_name = st.selectbox(
                "Select Conversion Model:",
                model_names,
                index=0
            )
            
            # Get model ID from name
            selected_model = available_models[model_names.index(selected_model_name)]
            
            st.markdown("---")
            
            # Page navigation
            page = st.radio(
                "Navigate to:",
                ["🔄 Text Converter", "📊 Model Performance", "ℹ️ Model Information", "🧪 Batch Processing"]
            )
            
            st.markdown("---")
            
            # Model info in sidebar
            if selected_model in self.model_metadata:
                st.subheader("Selected Model")
                st.write(f"**{self.model_metadata[selected_model]['model_name']}**")
                st.write(self.model_metadata[selected_model]['description'])

        # Main content based on page selection
        if page == "🔄 Text Converter":
            self.text_converter_page(selected_model)
        elif page == "📊 Model Performance":
            self.performance_page()
        elif page == "ℹ️ Model Information":
            self.model_info_page()
        elif page == "🧪 Batch Processing":
            self.batch_processing_page(selected_model)

    def text_converter_page(self, selected_model):
        """Text conversion interface"""
        st.header("🔄 Text Converter")
        
        # Input section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Input (Roman Urdu)")
            
            # Text input options
            input_method = st.radio("Input method:", ["Type text", "Use examples"])
            
            if input_method == "Type text":
                user_input = st.text_area(
                    "Enter Roman Urdu text:",
                    placeholder="Example: main acha hun, aap kaise hain?",
                    height=150
                )
            else:
                examples = [
                    "main acha hun",
                    "aap kaise hain",
                    "wo ghar ja raha hai",
                    "main kitab parh raha hun",
                    "aaj mausam bahut acha hai",
                    "hum school ja rahe hain"
                ]
                user_input = st.selectbox("Select example:", examples)
            
            # Convert button
            convert_button = st.button("🔄 Convert to Urdu", type="primary")
        
        with col2:
            st.subheader("📜 Output (Urdu Script)")
            
            if convert_button and user_input.strip():
                with st.spinner("Converting..."):
                    converted_text = self.convert_text(user_input, selected_model)
                
                # Display result
                st.markdown(f"""
                <div class="result-box">
                    <div class="urdu-text">{converted_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional info
                st.info(f"Converted using: {self.model_metadata.get(selected_model, {}).get('model_name', selected_model)}")
                
                # Word count
                input_words = len(user_input.split())
                output_words = len(converted_text.split())
                st.caption(f"Input: {input_words} words | Output: {output_words} words")
                
            elif convert_button:
                st.warning("Please enter some text to convert!")
            else:
                st.info("Enter text and click 'Convert to Urdu' to see the result.")
        
        # Compare models section
        st.markdown("---")
        st.subheader("🔬 Compare All Models")
        
        if st.button("Compare all available models"):
            if user_input and user_input.strip():
                comparison_results = {}
                
                for model_id in self.models.keys():
                    try:
                        result = self.convert_text(user_input, model_id)
                        comparison_results[model_id] = result
                    except Exception as e:
                        comparison_results[model_id] = f"Error: {e}"
                
                # Display comparison
                for model_id, result in comparison_results.items():
                    model_name = self.model_metadata.get(model_id, {}).get('model_name', model_id)
                    st.markdown(f"**{model_name}:**")
                    st.markdown(f'<div class="urdu-text" style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">{result}</div>', unsafe_allow_html=True)
                    st.markdown("")

    def performance_page(self):
        """Model performance visualization"""
        st.header("📊 Model Performance Analysis")
        
        if not self.evaluation_results:
            st.error("No evaluation results available!")
            return
        
        # Metrics overview
        st.subheader("📈 Performance Metrics Overview")
        
        # Prepare data for visualization
        models = []
        metrics_data = {}
        
        for model_key, metrics in self.evaluation_results.items():
            if model_key != 'test_info' and isinstance(metrics, dict):
                model_name = model_key.replace('_model', '').replace('_', ' ').title()
                models.append(model_name)
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(value)
        
        # Create performance comparison chart
        if models and metrics_data:
            # Bar chart for main metrics
            main_metrics = ['BLEU', 'ROUGE-L', 'Word_Accuracy', 'Character_Accuracy']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=main_metrics,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, metric in enumerate(main_metrics):
                if metric in metrics_data:
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=metrics_data[metric],
                            name=metric,
                            marker_color=colors[i],
                            text=[f"{v:.3f}" for v in metrics_data[metric]],
                            textposition='auto',
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics Table")
            
            df_data = []
            for i, model in enumerate(models):
                row = {'Model': model}
                for metric in metrics_data.keys():
                    if i < len(metrics_data[metric]):
                        row[metric] = f"{metrics_data[metric][i]:.3f}"
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Metrics explanation
            st.subheader("📚 Metrics Explanation")
            
            explanations = self.evaluation_results.get('test_info', {}).get('metrics_explanation', {})
            
            if explanations:
                for metric, explanation in explanations.items():
                    with st.expander(f"ℹ️ {metric}"):
                        st.write(explanation)
            
            # Best performing model
            st.subheader("🏆 Best Performing Model")
            
            if 'BLEU' in metrics_data:
                best_bleu_idx = np.argmax(metrics_data['BLEU'])
                best_model = models[best_bleu_idx]
                best_score = metrics_data['BLEU'][best_bleu_idx]
                
                st.success(f"**{best_model}** achieved the highest BLEU score: **{best_score:.3f}**")

    def model_info_page(self):
        """Model information and details"""
        st.header("ℹ️ Model Information")
        
        # Model cards
        for model_id, metadata in self.model_metadata.items():
            with st.container():
                st.markdown(f"""
                <div class="model-card">
                    <h3>{metadata.get('model_name', model_id)}</h3>
                    <p><strong>Type:</strong> {metadata.get('model_type', 'Unknown')}</p>
                    <p><strong>Description:</strong> {metadata.get('description', 'No description available')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Technical Details:**")
                    for key, value in metadata.items():
                        if key not in ['model_name', 'model_type', 'description']:
                            st.write(f"- {key}: {value}")
                
                with col2:
                    if model_id in self.evaluation_results:
                        st.write("**Performance Summary:**")
                        metrics = self.evaluation_results[model_id]
                        if isinstance(metrics, dict):
                            for metric, value in list(metrics.items())[:3]:  # Show top 3 metrics
                                if isinstance(value, (int, float)):
                                    st.write(f"- {metric}: {value:.3f}")
                
                st.markdown("---")

    def batch_processing_page(self, selected_model):
        """Batch processing interface"""
        st.header("🧪 Batch Processing")
        
        st.info("Process multiple Roman Urdu sentences at once")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Text area", "Upload file", "Use sample data"])
        
        sentences = []
        
        if input_method == "Text area":
            batch_text = st.text_area(
                "Enter Roman Urdu sentences (one per line):",
                placeholder="main acha hun\naap kaise hain\nwo ghar ja raha hai",
                height=200
            )
            sentences = [s.strip() for s in batch_text.split('\n') if s.strip()]
            
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                sentences = [s.strip() for s in content.split('\n') if s.strip()]
                
        elif input_method == "Use sample data":
            sample_sentences = [
                "main acha hun",
                "aap kaise hain",
                "wo ghar ja raha hai",
                "hum school ja rahe hain",
                "aaj mausam bahut acha hai",
                "main kitab parh raha hun",
                "wo khana kha raha hai",
                "hum cricket khel rahe hain"
            ]
            sentences = sample_sentences
            
            st.info(f"Using {len(sentences)} sample sentences")
        
        # Process button
        if st.button("🚀 Process Batch", type="primary") and sentences:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, sentence in enumerate(sentences):
                status_text.text(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
                
                try:
                    converted = self.convert_text(sentence, selected_model)
                    results.append({
                        'Input (Roman)': sentence,
                        'Output (Urdu)': converted,
                        'Status': 'Success'
                    })
                except Exception as e:
                    results.append({
                        'Input (Roman)': sentence,
                        'Output (Urdu)': f"Error: {e}",
                        'Status': 'Error'
                    })
                
                progress_bar.progress((i + 1) / len(sentences))
            
            status_text.text("Processing complete!")
            
            # Display results
            st.subheader("📊 Batch Processing Results")
            
            # Summary
            successful = sum(1 for r in results if r['Status'] == 'Success')
            st.metric("Successfully Processed", f"{successful}/{len(results)}")
            
            # Results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="roman_urdu_conversion_results.csv",
                mime="text/csv"
            )

def main():
    """Main function to run the Streamlit app"""
    try:
        converter = StreamlitRomanUrduConverter()
        converter.run()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please ensure all models are properly saved by running `python save_models.py` first.")

if __name__ == "__main__":
    main()
