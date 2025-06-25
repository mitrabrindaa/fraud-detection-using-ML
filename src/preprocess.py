import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

# Configure paths - UPDATED TO MATCH YOUR ACTUAL FILE NAME
BASE_DIR = Path(__file__).parent.parent
RAW_DATA = BASE_DIR / 'data/raw/onlinefraud.csv'  # Changed from transactions.csv
PROCESSED_DATA = BASE_DIR / 'data/processed/cleaned_transactions.csv'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=BASE_DIR / 'logs/data_processing.log'
)
logger = logging.getLogger(__name__)

def validate_input_file(filepath):
    """Validate the input file exists and is readable"""
    if not filepath.exists():
        raise FileNotFoundError(f"Input file {filepath} not found. Please ensure:"
                              f"\n1. The file exists at {filepath}"
                              f"\n2. Your working directory is {BASE_DIR}")
    if filepath.suffix != '.csv':
        raise ValueError("Only CSV files are supported")
    return True

def preprocess_data(input_file=RAW_DATA, output_file=PROCESSED_DATA):
    """
    Preprocess transaction data with robust error handling
    """
    try:
        validate_input_file(input_file)
        logger.info(f"Processing data from {input_file}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        chunks = pd.read_csv(input_file, chunksize=100000)
        processed_chunks = []
        encoder = LabelEncoder()
        
        for i, chunk in enumerate(chunks, 1):
            chunk = chunk.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], 
                             axis=1, errors='ignore')
            chunk['type'] = encoder.fit_transform(chunk['type'].astype(str))
            chunk.fillna(0, inplace=True)
            processed_chunks.append(chunk)
            logger.info(f"Processed chunk {i}")
        
        final_df = pd.concat(processed_chunks)
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        print(f"✅ Successfully processed data. Output at: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Now using the configured paths automatically
    preprocess_data()