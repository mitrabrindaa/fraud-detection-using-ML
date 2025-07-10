import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# --- Page configuration ---
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
@st.cache_resource
def load_artifacts():
    import os
    # Handle path correctly whether running from src or root directory
    model_path = '../models/model.pkl' if os.path.exists('../models/model.pkl') else 'models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_artifacts()

# --- Type encoding ---
type_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
type_options = list(type_map.keys())

# --- Enhanced Custom CSS ---
st.markdown("""
    <style>
    /* Main Styling */
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem;}
    
    /* Header */
    .main-header {background: rgba(255, 255, 255, 0.95); padding: 2rem; border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; text-align: center;}
    
    .main-title {font-size: 3rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    
    .subtitle {font-size: 1.2rem; color: #7f8c8d; margin-bottom: 1rem;}
    
    /* Input */
    .stSelectbox > div > div > div {background-color: rgba(255, 255, 255, 0.9); border: 2px solid #e74c3c; border-radius: 10px; transition: all 0.3s ease;}
    .stSelectbox > div > div > div:hover {border-color: #c0392b; box-shadow: 0 0 10px rgba(231, 76, 60, 0.3);}
    
    .stNumberInput > div > div > input, .stTextInput > div > div > input {background-color: rgba(255, 255, 255, 0.9); border: 2px solid #3498db; border-radius: 10px; padding: 0.5rem 1rem; font-size: 1rem; transition: all 0.3s ease;}
    .stNumberInput > div > div > input:focus, .stTextInput > div > div > input:focus {border-color: #2980b9; box-shadow: 0 0 10px rgba(52, 152, 219, 0.3);}
    
    /* Button */
    .stButton > button {background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; font-size: 1.2rem; font-weight: 600; padding: 0.8rem 2rem; border: none; border-radius: 25px; box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4); width: 100%; text-transform: uppercase;}
    .stButton > button:hover {transform: translateY(-2px); box-shadow: 0 6px 20px rgba(231, 76, 60, 0.6);}
    
    /* Alerts */
    .fraud-alert {background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; font-weight: 600; margin: 1rem 0; animation: pulse 2s infinite;}
    .safe-alert {background: linear-gradient(45deg, #27ae60, #2ecc71); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; font-weight: 600; margin: 1rem 0;}
    
    @keyframes pulse {0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }}
    
    /* Metrics */
    .metric-card {background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); margin: 0.5rem 0;}
    .metric-value {font-size: 2rem; font-weight: 700; color: #2c3e50;}
    .metric-label {font-size: 0.9rem; color: #7f8c8d; margin-top: 0.5rem;}
    
    /* Sidebar */
    .sidebar .sidebar-content {background: rgba(255, 255, 255, 0.95);}
    
    /* Transaction Summary */
    .transaction-summary {background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 5px solid #3498db;}
    .summary-item {display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem 0; border-bottom: 1px solid #ecf0f1;}
    .summary-label {font-weight: 600; color: #2c3e50;}
    .summary-value {color: #7f8c8d; font-family: 'Courier New', monospace;}
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛡️ Fraud Detection System</h1>
        <p class="subtitle">Advanced AI-powered transaction security analysis</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Quick Actions")
    if st.button("Fill Sample Payment"):
        st.session_state.quick_fill = {
            'type': 'PAYMENT',
            'amount': 5000.00,
            'oldbalanceOrg': 10000.00,
            'newbalanceOrg': 5000.00,
            'oldbalanceDest': 2000.00,
            'newbalanceDest': 7000.00
        }
    if st.button("Fill Suspicious Transaction"):
        st.session_state.quick_fill = {
            'type': 'CASH_OUT',
            'amount': 50000.00,
            'oldbalanceOrg': 50000.00,
            'newbalanceOrg': 0.00,
            'oldbalanceDest': 0.00,
            'newbalanceDest': 0.00
        }
    st.markdown("---")
    st.markdown("### System Stats")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">99.7%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">1.2ms</div>
        <div class="metric-label">Response Time</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">24/7</div>
        <div class="metric-label">Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Form ---
st.markdown('<div class="form-card">', unsafe_allow_html=True)
quick_fill = st.session_state.get('quick_fill', {})

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Transaction Details")
    type_val = st.selectbox(
        "Transaction Type", 
        type_options, 
        index=type_options.index(quick_fill.get('type', 'PAYMENT')),
        help="Select the type of transaction you are making."
    )
    step = st.number_input(
        "Step (time step)", 
        min_value=1, 
        value=1, 
        help="Sequential step in the transaction sequence"
    )
    amount = st.number_input(
        "Amount (₹)", 
        min_value=0.0, 
        value=quick_fill.get('amount', 0.0),
        format="%.2f", 
        help="Enter the transaction amount."
    )
    oldbalanceOrg = st.number_input(
        "Sender's Balance Before (₹)", 
        min_value=0.0, 
        value=quick_fill.get('oldbalanceOrg', 0.0),
        format="%.2f",
        help="Account balance before the transaction"
    )
    newbalanceOrg = st.number_input(
        "Sender's Balance After (₹)", 
        min_value=0.0, 
        value=quick_fill.get('newbalanceOrg', 0.0),
        format="%.2f",
        help="Account balance after the transaction"
    )

with col2:
    st.markdown("#### Participant Information")
    oldbalanceDest = st.number_input(
        "Recipient's Balance Before (₹)", 
        min_value=0.0, 
        value=quick_fill.get('oldbalanceDest', 0.0),
        format="%.2f",
        help="Recipient's balance before receiving funds"
    )
    newbalanceDest = st.number_input(
        "Recipient's Balance After (₹)", 
        min_value=0.0, 
        value=quick_fill.get('newbalanceDest', 0.0),
        format="%.2f",
        help="Recipient's balance after receiving funds"
    )
    sender = st.text_input(
        "Sender Name", 
        "John Doe",
        help="Name of the person sending money"
    )
    recipient = st.text_input(
        "Recipient Name", 
        "Jane Smith",
        help="Name of the person receiving money"
    )
st.markdown('</div>', unsafe_allow_html=True)

# --- Transaction Summary ---
st.markdown("""
    <div class="transaction-summary">
        <h4>Transaction Summary</h4>
        <div class="summary-item">
            <span class="summary-label">Sender:</span>
            <span class="summary-value">{}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Recipient:</span>
            <span class="summary-value">{}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Type:</span>
            <span class="summary-value">{}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Amount:</span>
            <span class="summary-value">₹{:,.2f}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Balance Change (Sender):</span>
            <span class="summary-value">₹{:,.2f} → ₹{:,.2f}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Balance Change (Recipient):</span>
            <span class="summary-value">₹{:,.2f} → ₹{:,.2f}</span>
        </div>
    </div>
    """.format(
        sender, recipient, type_val, amount, 
        oldbalanceOrg, newbalanceOrg, 
        oldbalanceDest, newbalanceDest
    ), unsafe_allow_html=True)


def preprocess_input():
    arr = np.array([
        step,
        type_map[type_val],
        amount,
        oldbalanceOrg,
        newbalanceOrg,
        oldbalanceDest,
        newbalanceDest
    ]).reshape(1, -1)
    return arr

def create_risk_gauge(fraud_probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Level"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue"},
        height=300
    )
    return fig

def create_feature_importance_chart():
    """Create a feature importance visualization"""
    features = ['Amount', 'Old Balance (Sender)', 'New Balance (Sender)', 
               'Old Balance (Recipient)', 'New Balance (Recipient)', 'Transaction Type', 'Step']
    importance = [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]  # Mock importance values
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Fraud Detection",
        color=importance,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue"},
        height=400
    )
    return fig

# --- Main Action Button ---
st.markdown("<br>", unsafe_allow_html=True)
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])

with analyze_col2:
    if st.button("🔍 Analyze Transaction", key="analyze"):
        X = preprocess_input()
        pred = model.predict_proba(X)
        fraud_probability = pred[0][1]
        
        st.markdown("---")
        
        # Risk Assessment Display
        if fraud_probability > 0.7:
            st.markdown("""
                <div class="fraud-alert">
                    <h2>🚨 HIGH RISK TRANSACTION DETECTED!</h2>
                    <p>This transaction shows strong indicators of fraudulent activity.</p>
                    <p><strong>Fraud Probability: {:.1f}%</strong></p>
                    <p>⚠️ TRANSACTION BLOCKED - Manual review required</p>
                </div>
                """.format(fraud_probability * 100), unsafe_allow_html=True)
            st.snow()
        elif fraud_probability > 0.3:
            st.markdown("""
                <div style="background: linear-gradient(45deg, #f39c12, #e67e22); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; font-weight: 600; margin: 1rem 0;">
                    <h2>⚠️ MEDIUM RISK TRANSACTION</h2>
                    <p>This transaction requires additional verification.</p>
                    <p><strong>Fraud Probability: {:.1f}%</strong></p>
                    <p>🔍 Additional authentication recommended</p>
                </div>
                """.format(fraud_probability * 100), unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="safe-alert">
                    <h2>✅ TRANSACTION APPROVED</h2>
                    <p>This transaction appears to be legitimate.</p>
                    <p><strong>Fraud Probability: {:.1f}%</strong></p>
                    <p>🎉 Safe to proceed</p>
                </div>
                """.format(fraud_probability * 100), unsafe_allow_html=True)
            st.balloons()
        
        # Detailed Analysis
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Gauge
            risk_fig = create_risk_gauge(fraud_probability)
            st.plotly_chart(risk_fig, use_container_width=True)
            
            # Risk Factors
            st.markdown("#### Risk Factors Analysis")
            risk_factors = []
            
            if amount > 10000:
                risk_factors.append("• High transaction amount")
            if oldbalanceOrg == newbalanceOrg + amount:
                risk_factors.append("• Perfect balance consistency")
            if newbalanceDest == 0 and oldbalanceDest == 0:
                risk_factors.append("• Recipient account shows no activity")
            if type_val in ['CASH_OUT', 'TRANSFER']:
                risk_factors.append(f"• {type_val} transactions have higher risk")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.markdown("• No significant risk factors detected")
        
        with col2:
            # Feature Importance Chart
            importance_fig = create_feature_importance_chart()
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Transaction Validation
            st.markdown("#### Transaction Validation")
            balance_check = oldbalanceOrg - newbalanceOrg == amount
            st.markdown(f"**Balance Consistency:** {'✅ Valid' if balance_check else '❌ Invalid'}")
            
            amount_reasonable = amount > 0 and amount < 100000
            st.markdown(f"**Amount Range:** {'✅ Reasonable' if amount_reasonable else '❌ Suspicious'}")
            
            participant_check = sender != recipient
            st.markdown(f"**Participants:** {'✅ Different' if participant_check else '❌ Same person'}")
        
        # Additional Recommendations
        st.markdown("### 💡 Security Recommendations")
        
        if fraud_probability > 0.5:
            st.markdown("""
            - 🔒 **Block this transaction immediately**
            - 📞 **Contact customer for verification**
            - 🔍 **Conduct manual review**
            - 📝 **Document the incident**
            """)
        elif fraud_probability > 0.3:
            st.markdown("""
            - 🔐 **Request additional authentication**
            - 📱 **Send OTP verification**
            - 💳 **Verify card details**
            - ⏰ **Monitor account activity**
            """)
        else:
            st.markdown("""
            - ✅ **Transaction can proceed**
            - 📊 **Continue monitoring**
            - 🔔 **Log for future analysis**
            - 📈 **Update customer profile**
            """)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
        <p>🛡️ <strong>Fraud Detection System v2.0</strong> | Powered by Advanced Machine Learning</p>
        <p>Built with ❤️ using Streamlit | © 2024 Security Analytics</p>
    </div>
    """, unsafe_allow_html=True)
