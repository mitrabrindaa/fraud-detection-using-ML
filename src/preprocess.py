import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file='onlinefraud.csv', output_file='cleaned_transactions.csv'):
    # Load data in chunks for memory efficiency
    chunks = pd.read_csv(input_file, chunksize=100000)
    processed_chunks = []
    
    for chunk in chunks:
        # Drop non-predictive columns
        chunk = chunk.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
        
        # Encode categorical 'type' column
        le = LabelEncoder()
        chunk['type'] = le.fit_transform(chunk['type'].astype(str))
        
        # Handle missing values
        chunk.fillna(0, inplace=True)
        
        processed_chunks.append(chunk)
    
    # Combine and save
    pd.concat(processed_chunks).to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data()  # Uses 'onlinefraud.csv' by default
    