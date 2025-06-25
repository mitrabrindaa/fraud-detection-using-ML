import sys
import subprocess
import streamlit as st
from pathlib import Path
import logging
import pandas as pd

# ========== CONFIGURATION ==========
class AppConfig:
    """Centralized configuration for the app"""
    REQUIRED_PACKAGES = {
        'joblib': '1.3.2',
        'xgboost': '2.0.3',
        'scikit-learn': '1.3.2',
        'pandas': '2.1.4'
    }
    
    TYPE_MAP = {
        'PAYMENT': 0, 
        'TRANSFER': 1,
        'CASH_OUT': 2,
        'DEBIT': 3,
        'CASH_IN': 4
    }

# ========== INITIALIZATION ==========
def initialize_app():
    """Handle all startup operations"""
    install_dependencies()
    setup_paths()
    configure_logging()

def install_dependencies():
    """Ensure all required packages are installed"""
    for package, version in AppConfig.REQUIRED_PACKAGES.items():
        try:
            __import__(package)
        except ImportError:
            st.warning(f"⚠️ Installing {package}...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                f"{package}=={version}"
            ])

def setup_paths():
    """Configure and validate all file paths"""
    try:
        # Base directory setup
        global BASE_DIR, MODEL_PATH, DATA_DIR
        BASE_DIR = Path(__file__).parent.parent
        
        # Critical directories
        MODEL_PATH = BASE_DIR / 'models/fraud_model_latest.pkl'
        DATA_DIR = BASE_DIR / 'data/uploads'
        
        # Create directories with parents
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / 'logs').mkdir(exist_ok=True)
        
        # Validation
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
            
    except Exception as e:
        st.error(f"❌ Critical startup error: {str(e)}")
        st.stop()

def configure_logging():
    """Set up application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=BASE_DIR / 'logs/app.log'
    )
    global logger
    logger = logging.getLogger(__name__)

# ========== CORE FUNCTIONALITY ==========
def load_model():
    """Load the trained ML model"""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("❌ Failed to load fraud detection model")
        st.stop()

# ========== APP INITIALIZATION ==========
initialize_app()
model = load_model()

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# [Your existing Streamlit UI code goes here]
# Keep all your tab definitions and UI logic unchanged