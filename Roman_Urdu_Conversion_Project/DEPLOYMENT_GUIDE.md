# Streamlit Deployment Guide - Roman Urdu Converter

## 🚀 **Deployment Steps for Streamlit Cloud**

### **Step 1: Prepare Repository**

1. **Ensure these files are in your root directory:**
   ```
   streamlit_app.py          # Main application
   requirements_streamlit.txt # Dependencies
   packages.txt             # System packages (if needed)
   .streamlit/config.toml   # Streamlit configuration
   ```

### **Step 2: Fix Dependencies**

**Use the updated `requirements_streamlit.txt`:**
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
nltk>=3.8.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.20.0
arabic-reshaper>=2.1.0
python-bidi>=0.4.2
pathlib2>=2.3.0
requests>=2.28.0
```

### **Step 3: Repository Setup**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Fix deployment dependencies"
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select the main branch
   - Set main file: `streamlit_app.py`

### **Step 4: Advanced Configuration**

**If deployment still fails, try these minimal requirements:**

**Create `requirements_minimal.txt`:**
```
streamlit==1.28.0
pandas==1.5.0
numpy==1.21.0
plotly==5.15.0
scikit-learn==1.2.0
```

## 🔧 **Troubleshooting Common Issues**

### **Error: "installer returned a non-zero exit code"**

**Solutions:**

1. **Update pip in requirements:**
   ```
   # Add to top of requirements_streamlit.txt
   pip>=23.0.0
   ```

2. **Use specific versions instead of ranges:**
   ```
   streamlit==1.28.0
   pandas==1.5.3
   numpy==1.24.3
   ```

3. **Remove problematic packages:**
   - Remove `torch` if not essential
   - Remove `transformers` if not used
   - Use only core packages

### **Error: "Module not found"**

**Fix import paths in streamlit_app.py:**
```python
# Add error handling for imports
try:
    from models.dictionary_model import DictionaryModel
    from models.ml_model import MLModel
    from utils.preprocessing import RomanUrduPreprocessor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()
```

### **Error: "File not found"**

**Ensure all required files exist:**
```python
# Add file existence checks
if not os.path.exists("data/roman_urdu_dictionary.json"):
    st.error("Dictionary file not found")
    st.stop()

if not os.path.exists("models/saved/"):
    st.error("Saved models directory not found")
    st.stop()
```

## 🔄 **Alternative Deployment Methods**

### **Method 1: Local Streamlit (Testing)**
```bash
cd Roman_Urdu_Conversion_Project
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

### **Method 2: Heroku Deployment**

**Create `Procfile`:**
```
web: sh setup.sh && streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

**Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### **Method 3: Railway Deployment**

**Create `railway.toml`:**
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run streamlit_app.py --server.headless=true --server.port=$PORT"
```

## 📋 **Deployment Checklist**

- [ ] ✅ All imports are working locally
- [ ] ✅ requirements_streamlit.txt has correct versions
- [ ] ✅ No conflicting dependencies
- [ ] ✅ All model files are committed to repo
- [ ] ✅ Data files are included
- [ ] ✅ Config files are properly set
- [ ] ✅ Repository is public or accessible
- [ ] ✅ Branch is specified correctly
- [ ] ✅ Main file path is correct

## 🎯 **Quick Fix Commands**

**Local testing:**
```bash
# Update pip first
pip install --upgrade pip

# Install clean requirements
pip install -r requirements_streamlit.txt

# Test locally
streamlit run streamlit_app.py
```

**Repository update:**
```bash
# Commit all changes
git add .
git commit -m "Fix deployment issues"
git push origin main

# Force refresh deployment
# Go to Streamlit Cloud and restart app
```

## 🆘 **Emergency Minimal App**

If all else fails, create a minimal version:

**Create `streamlit_minimal.py`:**
```python
import streamlit as st
import pandas as pd
import numpy as np

st.title("🔤 Roman Urdu to Urdu Converter")
st.write("Minimal version for deployment testing")

text_input = st.text_input("Enter Roman Urdu:")
if text_input:
    st.write(f"Input: {text_input}")
    st.success("Basic app is working!")
```

## 📞 **Support Resources**

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-cloud
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: Check for similar deployment issues

**Your deployment should now work! 🚀**
