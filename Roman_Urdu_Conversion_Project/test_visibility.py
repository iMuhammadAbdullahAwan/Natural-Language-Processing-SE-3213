#!/usr/bin/env python3
"""
Test script to verify text visibility and styling in Streamlit app
"""
import streamlit as st

st.set_page_config(
    page_title="Text Visibility Test",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .test-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
        color: #212529;
    }
    
    .urdu-text-test {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif;
        font-size: 1.4em;
        text-align: right;
        direction: rtl;
        color: #1a1a1a;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Text Visibility Test")

st.header("1. Regular Text")
st.write("This is regular text - it should be dark on light background")

st.header("2. Roman Urdu Input")
test_input = st.text_input("Type Roman Urdu here:", value="main acha hun")

st.header("3. Urdu Output")
st.markdown(f"""
<div class="test-section">
    <h4>Test Urdu Text:</h4>
    <div class="urdu-text-test">میں اچھا ہوں</div>
</div>
""", unsafe_allow_html=True)

st.header("4. Model Cards Test")
st.markdown("""
<div class="test-section">
    <h4>Model Information</h4>
    <p>This text should be dark and clearly visible on light background.</p>
    <ul>
        <li>Dictionary Model: Working</li>
        <li>ML Model: Working</li>
        <li>Character Model: Working</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.header("5. Success/Info Messages")
st.success("✅ This success message should be green with dark text")
st.info("ℹ️ This info message should be blue with dark text")
st.warning("⚠️ This warning should be yellow with dark text")

st.header("6. Metrics Display")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("BLEU Score", "0.745", "0.12")
with col2:
    st.metric("Word Accuracy", "82.5%", "5.2%")
with col3:
    st.metric("Character Accuracy", "91.2%", "3.8%")

st.header("7. Data Display")
import pandas as pd
test_data = pd.DataFrame({
    'Model': ['Dictionary', 'Word ML', 'Char ML'],
    'BLEU': [0.745, 0.712, 0.689],
    'Accuracy': [0.825, 0.798, 0.776]
})
st.dataframe(test_data)

st.markdown("---")
st.markdown("**Test Summary:** All text elements above should be clearly visible with proper contrast.")
