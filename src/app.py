import sys
import subprocess
import streamlit as st

# Dependency enforcement block
REQUIRED_PACKAGES = {
    'joblib': '1.3.2',
    'xgboost': '2.0.3',
    'scikit-learn': '1.3.2'
}

def install_package(package, version):
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        f"{package}=={version}"
    ])

for package, version in REQUIRED_PACKAGES.items():
    try:
        __import__(package)
    except ImportError:
        st.warning(f"⚠️ Installing {package}...")
        install_package(package, version)

# Now safe to import
import pandas as pd
import joblib
import logging
from pathlib import Path

# Force install joblib if missing (for Streamlit Cloud)
try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.3.2"])
    import joblib

# Configure paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models/fraud_model_latest.pkl'
DATA_DIR = BASE_DIR / 'data/uploads'
DATA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=BASE_DIR / 'logs/app.log'
)
logger = logging.getLogger(__name__)

# Type mapping (must match preprocess.py encoding)
TYPE_MAP = {
    'PAYMENT': 0, 
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("❌ Failed to load fraud detection model")
        st.stop()

# ========== Streamlit UI Code Below ==========
# [Keep your existing Streamlit UI code unchanged]
