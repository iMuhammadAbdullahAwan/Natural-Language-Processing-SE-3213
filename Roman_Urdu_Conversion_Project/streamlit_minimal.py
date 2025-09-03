import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Roman Urdu to Urdu Converter",
    page_icon="🔤",
    layout="wide"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2E86AB !important;
    }
    
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
    }
</style>
""", unsafe_allow_html=True)

def load_dictionary():
    """Load a basic dictionary for conversion"""
    try:
        dict_path = Path("data/roman_urdu_dictionary.json")
        if dict_path.exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('dictionary', {})
        else:
            # Fallback basic dictionary
            return {
                "main": "میں",
                "aap": "آپ",
                "kaise": "کیسے",
                "hain": "ہیں",
                "theek": "ٹھیک",
                "hun": "ہوں",
                "acha": "اچھا",
                "hai": "ہے",
                "wo": "وہ",
                "ghar": "گھر",
                "ja": "جا",
                "raha": "رہا",
                "rahe": "رہے",
                "school": "سکول",
                "aaj": "آج",
                "mausam": "موسم",
                "bahut": "بہت"
            }
    except Exception as e:
        st.error(f"Error loading dictionary: {e}")
        return {}

def simple_convert(text, dictionary):
    """Simple word-by-word conversion"""
    if not text:
        return ""
    
    words = text.lower().split()
    converted_words = []
    
    for word in words:
        converted_word = dictionary.get(word, word)
        converted_words.append(converted_word)
    
    return ' '.join(converted_words)

def main():
    st.markdown('<h1 style="text-align: center;">🔤 Roman Urdu to Urdu Script Converter</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Convert Roman Urdu text to Urdu script</p>', unsafe_allow_html=True)
    
    # Load dictionary
    dictionary = load_dictionary()
    
    if not dictionary:
        st.warning("⚠️ Dictionary not loaded. Using basic conversion.")
    else:
        st.success(f"✅ Dictionary loaded with {len(dictionary)} words")
    
    # Main conversion interface
    st.header("📝 Text Conversion")
    
    # Input text
    input_text = st.text_area(
        "Enter Roman Urdu text:",
        placeholder="Type Roman Urdu here... (e.g., main acha hun)",
        height=100
    )
    
    if input_text:
        # Convert text
        converted = simple_convert(input_text, dictionary)
        
        # Display result
        st.subheader("🔄 Converted Text:")
        st.markdown(f'<div class="urdu-text">{converted}</div>', unsafe_allow_html=True)
        
        # Show word-by-word breakdown
        with st.expander("📋 Word-by-word breakdown"):
            words = input_text.lower().split()
            converted_words = converted.split()
            
            breakdown_data = {
                'Roman Urdu': words,
                'Urdu Script': converted_words[:len(words)]
            }
            
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True)
    
    # Sample examples
    st.header("📚 Try These Examples")
    
    examples = [
        "main acha hun",
        "aap kaise hain",
        "wo ghar ja raha hai",
        "aaj mausam bahut acha hai"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, example in enumerate(examples[:2]):
            if st.button(f"Try: {example}", key=f"example_{i}"):
                st.session_state.example_text = example
    
    with col2:
        for i, example in enumerate(examples[2:], 2):
            if st.button(f"Try: {example}", key=f"example_{i}"):
                st.session_state.example_text = example
    
    # Handle example selection
    if hasattr(st.session_state, 'example_text'):
        converted_example = simple_convert(st.session_state.example_text, dictionary)
        st.markdown("**Example conversion:**")
        st.write(f"Roman: {st.session_state.example_text}")
        st.markdown(f'<div class="urdu-text">{converted_example}</div>', unsafe_allow_html=True)
    
    # Dictionary stats
    st.header("📊 Dictionary Statistics")
    if dictionary:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", len(dictionary))
        with col2:
            st.metric("Status", "Loaded ✅")
        with col3:
            st.metric("Coverage", "Basic")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.write("""
        **Roman Urdu to Urdu Script Converter**
        
        This is a simplified version of the Roman Urdu to Urdu conversion system.
        
        **Features:**
        - Dictionary-based word conversion
        - Real-time text processing
        - Word-by-word breakdown
        - Sample examples
        
        **How it works:**
        1. Enter Roman Urdu text in the input area
        2. The system converts each word using the dictionary
        3. Unknown words remain unchanged
        4. View the Urdu script result
        
        **Note:** This is a basic version designed for deployment compatibility.
        """)

if __name__ == "__main__":
    main()
