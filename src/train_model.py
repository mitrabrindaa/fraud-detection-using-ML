import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging
from pathlib import Path
from datetime import datetime

# Configure paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA = BASE_DIR / 'data/processed/cleaned_transactions.csv'
MODELS_DIR = BASE_DIR / 'models'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=BASE_DIR / 'logs/training.log'
)
logger = logging.getLogger(__name__)

def load_data(filepath=PROCESSED_DATA):
    """Load and validate training data"""
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"Data file {filepath} not found")
        
        df = pd.read_csv(filepath)
        if 'isFraud' not in df.columns:
            raise ValueError("Target column 'isFraud' missing in data")
            
        logger.info(f"Loaded data from {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def train_model():
    """Main training pipeline"""
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        
        df = load_data()
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = XGBClassifier(
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric='auc',
            random_state=42,
            early_stopping_rounds=10
        )
        
        logger.info("Starting model training...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        
        # Save model with timestamp AND a static reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f'fraud_model_{timestamp}.pkl'
        static_path = MODELS_DIR / 'fraud_model_latest.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(model, static_path)  # For app.py to reference
        
        logger.info(f"Model saved to {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()