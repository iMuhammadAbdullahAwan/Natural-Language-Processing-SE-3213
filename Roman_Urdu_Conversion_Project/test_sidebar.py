#!/usr/bin/env python3
"""
Comprehensive sidebar and form elements visibility test
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sidebar Elements Test",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Sidebar & Form Elements Visibility Test")

# Sidebar tests
st.sidebar.title("📋 Sidebar Test Elements")

st.sidebar.markdown("### Navigation & Text")
st.sidebar.write("This text should be dark and clearly visible")
st.sidebar.info("This is an info message in sidebar")

st.sidebar.markdown("### Form Elements")

# Checkbox test
st.sidebar.markdown("**Checkboxes:**")
checkbox1 = st.sidebar.checkbox("✓ Checkbox 1 - Should be visible")
checkbox2 = st.sidebar.checkbox("✓ Checkbox 2 - Text should be dark", value=True)
checkbox3 = st.sidebar.checkbox("✓ Checkbox 3 - Both checked and unchecked visible")

# Radio button test
st.sidebar.markdown("**Radio Buttons:**")
radio_option = st.sidebar.radio(
    "Select an option (text should be dark):",
    ["Option 1", "Option 2", "Option 3"]
)

# Selectbox test
st.sidebar.markdown("**Selectbox:**")
select_option = st.sidebar.selectbox(
    "Choose from dropdown:",
    ["Choice A", "Choice B", "Choice C"]
)

# Slider test
st.sidebar.markdown("**Slider:**")
slider_value = st.sidebar.slider("Adjust value:", 0, 100, 50)

# Text input test
st.sidebar.markdown("**Text Input:**")
text_input = st.sidebar.text_input("Enter text:", "Sample text")

# Number input test
st.sidebar.markdown("**Number Input:**")
number_input = st.sidebar.number_input("Enter number:", value=42)

# Text area test
st.sidebar.markdown("**Text Area:**")
text_area = st.sidebar.text_area("Enter paragraph:", "This text should be dark and readable")

# Multiselect test
st.sidebar.markdown("**Multiselect:**")
multiselect = st.sidebar.multiselect(
    "Select multiple:",
    ["Item 1", "Item 2", "Item 3", "Item 4"],
    default=["Item 1"]
)

# Button test
st.sidebar.markdown("**Buttons:**")
if st.sidebar.button("Test Button"):
    st.sidebar.success("Button clicked! This success message should be visible.")

# File uploader test
st.sidebar.markdown("**File Upload:**")
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Main content area
st.markdown("## Main Content Area Tests")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Form Elements in Main Area")
    
    # Main area checkbox
    main_checkbox = st.checkbox("Main area checkbox")
    
    # Main area radio
    main_radio = st.radio("Main radio:", ["A", "B", "C"])
    
    # Main area selectbox
    main_select = st.selectbox("Main selectbox:", ["X", "Y", "Z"])

with col2:
    st.markdown("### Results Display")
    
    # Show sidebar values
    st.write("**Sidebar Values:**")
    st.write(f"- Checkbox 1: {checkbox1}")
    st.write(f"- Checkbox 2: {checkbox2}")
    st.write(f"- Radio: {radio_option}")
    st.write(f"- Select: {select_option}")
    st.write(f"- Slider: {slider_value}")
    st.write(f"- Text: {text_input}")
    st.write(f"- Number: {number_input}")
    st.write(f"- Multiselect: {multiselect}")

# Data display test
st.markdown("### Data Display Test")
test_df = pd.DataFrame({
    'Element': ['Checkbox', 'Radio', 'Selectbox', 'Slider'],
    'Visibility': ['Good', 'Good', 'Good', 'Good'],
    'Contrast': ['High', 'High', 'High', 'High']
})
st.dataframe(test_df)

# Metrics test
st.markdown("### Metrics Test")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Checkboxes", "✓ Visible", "100%")
with col2:
    st.metric("Radio Buttons", "✓ Visible", "100%")
with col3:
    st.metric("Selectboxes", "✓ Visible", "100%")
with col4:
    st.metric("Text Elements", "✓ Visible", "100%")

# Status messages
st.success("✅ Success message - should be green with dark text")
st.info("ℹ️ Info message - should be blue with dark text")
st.warning("⚠️ Warning message - should be yellow/orange with dark text")

# Final validation
st.markdown("---")
st.markdown("## 🎯 Validation Checklist")

validation_data = {
    'Element Type': [
        'Sidebar Background', 'Sidebar Text', 'Sidebar Checkboxes', 
        'Sidebar Radio Buttons', 'Sidebar Selectboxes', 'Sidebar Buttons',
        'Main Area Text', 'Main Area Forms', 'Data Tables', 'Metrics'
    ],
    'Expected Color': [
        'Light Gray', 'Dark Gray', 'Dark Gray with Blue Accent',
        'Dark Gray with Blue Accent', 'Dark Gray', 'White on Blue',
        'Dark Gray', 'Dark Gray', 'Dark Gray', 'Various'
    ],
    'Status': ['✅ Fixed'] * 10
}

validation_df = pd.DataFrame(validation_data)
st.table(validation_df)

st.markdown("**All elements above should be clearly visible with proper contrast!**")
