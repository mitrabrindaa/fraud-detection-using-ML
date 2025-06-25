import streamlit as st
import pandas as pd
import joblib
from io import StringIO

# Load model
model = joblib.load('fraud_model.pkl')

# Type mapping
type_map = {
    'PAYMENT': 0, 
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}

# UI Config
st.set_page_config(layout="wide")
st.title("💳 AI Fraud Detection System")

# Tabbed interface
tab1, tab2 = st.tabs(["Single Check", "Batch Processing"])

with tab1:
    st.header("Real-Time Fraud Detection")
    with st.form("single_txn"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            old_bal_orig = st.number_input("Origin Old Balance", min_value=0.0)
            new_bal_orig = st.number_input("Origin New Balance", min_value=0.0)
        
        with col2:
            old_bal_dest = st.number_input("Destination Old Balance", min_value=0.0)
            new_bal_dest = st.number_input("Destination New Balance", min_value=0.0)
            txn_type = st.selectbox("Type", list(type_map.keys()))
        
        submitted = st.form_submit_button("Check Risk")

    if submitted:
        input_df = pd.DataFrame([[
            type_map[txn_type],
            amount,
            old_bal_orig,
            new_bal_orig,
            old_bal_dest,
            new_bal_dest
        ]], columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
        
        proba = model.predict_proba(input_df)[0][1] * 100
        
        if proba > 75:
            st.error(f"🚨 HIGH RISK ({proba:.1f}%)")
        elif proba > 25:
            st.warning(f"⚠️ MEDIUM RISK ({proba:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({proba:.1f}%)")

with tab2:
    st.header("Process CSV File")
    uploaded_file = st.file_uploader("Upload transactions CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Preprocess like training data
        df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore')
        df['type'] = df['type'].map(type_map)
        df.fillna(0, inplace=True)
        
        # Predict
        df['Fraud Probability'] = model.predict_proba(df.drop('isFraud', axis=1, errors='ignore'))[:,1] * 100
        df['Risk Level'] = pd.cut(
            df['Fraud Probability'],
            bins=[0, 25, 75, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Show results
        st.dataframe(df.sort_values('Fraud Probability', ascending=False))
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Results",
            csv,
            "fraud_results.csv",
            "text/csv"
        )