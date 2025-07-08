import streamlit as st
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")

import numpy as np
import pickle

# --- Load Model ---
@st.cache_resource
def load_artifacts():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_artifacts()

# --- Type encoding (adjust if your model uses different mapping) ---
type_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
type_options = list(type_map.keys())

# --- Streamlit UI ---
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px;}
    .stTextInput>div>div>input, .stNumberInput>div>div>input {background-color: #fffbe7;}
    </style>
    """, unsafe_allow_html=True)

st.title("💳 Online Transaction Fraud Detection")
st.markdown("""
#### Simulate a Transaction
Fill in the details below to simulate an online money transfer or payment. The system will instantly analyze your transaction for fraud risk.
""")

col1, col2 = st.columns(2)
with col1:
    type_val = st.selectbox("Transaction Type", type_options, help="Select the type of transaction you are making.")
    amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f", help="Enter the transaction amount.")
    oldbalanceOrg = st.number_input("Your Account Balance Before (₹)", min_value=0.0, format="%.2f")
    newbalanceOrg = st.number_input("Your Account Balance After (₹)", min_value=0.0, format="%.2f")
with col2:
    oldbalanceDest = st.number_input("Recipient's Balance Before (₹)", min_value=0.0, format="%.2f")
    newbalanceDest = st.number_input("Recipient's Balance After (₹)", min_value=0.0, format="%.2f")
    sender = st.text_input("Your Name", "John Doe")
    recipient = st.text_input("Recipient Name", "Jane Smith")

st.markdown("---")
st.markdown(f"**Sender:** {sender}  →  **Recipient:** {recipient}")
st.markdown(f"**Transaction:** {type_val}  |  **Amount:** ₹{amount:,.2f}")


def preprocess_input():
    arr = np.array([
        type_map[type_val],
        amount,
        oldbalanceOrg,
        newbalanceOrg,
        oldbalanceDest,
        newbalanceDest
    ]).reshape(1, -1)
    return arr

if st.button("Send Transaction 🚀"):
    X = preprocess_input()
    pred = model.predict_proba(X)
    st.markdown("---")
    st.success(f"Fraud Probability: {pred[0][1]*100:.2f}%")
    st.info("🚨 Fraudulent Transaction Detected!" if pred[0][1] > 0.5 else "✅ Transaction is Legitimate.")
    st.balloons() if pred[0][1] <= 0.5 else st.snow()
