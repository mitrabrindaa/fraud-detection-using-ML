#!/usr/bin/env python3
"""
Flask Web Application for Fraud Detection System
A professional web interface for the fraud detection model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)

# --- Load Model ---
def load_model():
    """Load the fraud detection model"""
    model_path = 'models/model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None

model = load_model()

# --- Transaction Type Mapping ---
type_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}

@app.route('/')
def index():
    """Main page with fraud detection form"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    """Analyze transaction for fraud"""
    try:
        data = request.json
        print("Received data:", data)  # Debugging log

        # Extract transaction data
        step = int(data.get('step', 1))
        transaction_type = data.get('type', 'PAYMENT')
        amount = float(data.get('amount', 0))
        old_balance_orig = float(data.get('oldbalanceOrg', 0))
        new_balance_orig = float(data.get('newbalanceOrg', 0))
        old_balance_dest = float(data.get('oldbalanceDest', 0))
        new_balance_dest = float(data.get('newbalanceDest', 0))
        sender = data.get('sender', '')
        recipient = data.get('recipient', '')
        
        # Prepare input for model
        X = np.array([
            step,
            type_map[transaction_type],
            amount,
            old_balance_orig,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest
        ]).reshape(1, -1)
        print("Model input array:", X)  # Debugging log
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction = model.predict_proba(X)
        print("Model prediction:", prediction)  # Debugging log
        fraud_probability = float(prediction[0][1])
        
        # Determine risk level
        if fraud_probability > 0.7:
            risk_level = 'HIGH'
            risk_color = '#e74c3c'
            recommendation = 'BLOCK TRANSACTION'
        elif fraud_probability > 0.3:
            risk_level = 'MEDIUM'
            risk_color = '#f39c12'
            recommendation = 'ADDITIONAL VERIFICATION'
        else:
            risk_level = 'LOW'
            risk_color = '#27ae60'
            recommendation = 'APPROVE TRANSACTION'
        
        # Analyze risk factors
        risk_factors = []
        if amount > 10000:
            risk_factors.append('High transaction amount')
        if old_balance_orig == new_balance_orig + amount:
            risk_factors.append('Perfect balance consistency')
        if new_balance_dest == 0 and old_balance_dest == 0:
            risk_factors.append('Recipient account shows no activity')
        if transaction_type in ['CASH_OUT', 'TRANSFER']:
            risk_factors.append(f'{transaction_type} transactions have higher risk')
        
        # Transaction validation
        balance_check = old_balance_orig - new_balance_orig == amount
        amount_reasonable = amount > 0 and amount < 100000
        participant_check = sender != recipient
        
        response = {
            'fraud_probability': fraud_probability * 100,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'validation': {
                'balance_consistent': balance_check,
                'amount_reasonable': amount_reasonable,
                'different_participants': participant_check
            },
            'transaction_details': {
                'sender': sender,
                'recipient': recipient,
                'type': transaction_type,
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print("Response data:", response)  # Debugging log
        return jsonify(response)
        
    except Exception as e:
        print("Error during analysis:", str(e))  # Debugging log
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-data')
def get_sample_data():
    """Get sample transaction data for testing with multiple variations"""
    samples = {
        'legitimate': [
            {
                'type': 'PAYMENT',
                'amount': 5000.00,
                'oldbalanceOrg': 10000.00,
                'newbalanceOrg': 5000.00,
                'oldbalanceDest': 2000.00,
                'newbalanceDest': 7000.00,
                'sender': 'John Doe',
                'recipient': 'Jane Smith',
                'description': 'Regular payment transaction'
            },
            {
                'type': 'TRANSFER',
                'amount': 2500.00,
                'oldbalanceOrg': 15000.00,
                'newbalanceOrg': 12500.00,
                'oldbalanceDest': 5000.00,
                'newbalanceDest': 7500.00,
                'sender': 'Alice Johnson',
                'recipient': 'Bob Wilson',
                'description': 'Bank transfer between accounts'
            },
            {
                'type': 'DEBIT',
                'amount': 750.00,
                'oldbalanceOrg': 8000.00,
                'newbalanceOrg': 7250.00,
                'oldbalanceDest': 1000.00,
                'newbalanceDest': 1750.00,
                'sender': 'Sarah Chen',
                'recipient': 'Online Store',
                'description': 'Debit card purchase'
            },
            {
                'type': 'CASH_IN',
                'amount': 1000.00,
                'oldbalanceOrg': 3000.00,
                'newbalanceOrg': 2000.00,
                'oldbalanceDest': 8000.00,
                'newbalanceDest': 9000.00,
                'sender': 'Michael Brown',
                'recipient': 'ATM Deposit',
                'description': 'Cash deposit at ATM'
            }
        ],
        'suspicious': [
            {
                'type': 'CASH_OUT',
                'amount': 85000.00,
                'oldbalanceOrg': 85000.00,
                'newbalanceOrg': 0.00,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 0.00,
                'sender': 'Anonymous User',
                'recipient': 'Unknown Account',
                'description': 'Large cash withdrawal with zero balances'
            },
            {
                'type': 'TRANSFER',
                'amount': 125000.00,
                'oldbalanceOrg': 130000.00,
                'newbalanceOrg': 5000.00,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 0.00,
                'sender': 'Fake Account',
                'recipient': 'Shell Company',
                'description': 'High-value transfer to inactive account'
            },
            {
                'type': 'CASH_OUT',
                'amount': 200000.00,
                'oldbalanceOrg': 200000.00,
                'newbalanceOrg': 0.00,
                'oldbalanceDest': 500000.00,
                'newbalanceDest': 500000.00,
                'sender': 'Suspicious Entity',
                'recipient': 'Money Mule',
                'description': 'Massive cash-out with no recipient balance change'
            },
            {
                'type': 'PAYMENT',
                'amount': 99999.99,
                'oldbalanceOrg': 100000.00,
                'newbalanceOrg': 0.01,
                'oldbalanceDest': 0.00,
                'newbalanceDest': 0.00,
                'sender': 'Test Account',
                'recipient': 'Dummy Recipient',
                'description': 'Maximum amount payment to inactive account'
            }
        ],
        'mixed': [
            {
                'type': 'TRANSFER',
                'amount': 25000.00,
                'oldbalanceOrg': 50000.00,
                'newbalanceOrg': 25000.00,
                'oldbalanceDest': 10000.00,
                'newbalanceDest': 35000.00,
                'sender': 'Corporate Account',
                'recipient': 'Business Partner',
                'description': 'Medium risk corporate transfer'
            },
            {
                'type': 'CASH_OUT',
                'amount': 15000.00,
                'oldbalanceOrg': 20000.00,
                'newbalanceOrg': 5000.00,
                'oldbalanceDest': 2000.00,
                'newbalanceDest': 2000.00,
                'sender': 'Regular Customer',
                'recipient': 'ATM Network',
                'description': 'Large cash withdrawal from regular account'
            }
        ]
    }
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
