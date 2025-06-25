import streamlit as st
import pandas as pd
import joblib
import logging
from pathlib import Path

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

# ... [rest of your existing app.py code remains the same]