# 🛡️ Fraud Detection Web Application

A professional web interface for the fraud detection system built with Flask and Bootstrap.

## ✨ Features

- **Professional UI**: Modern Bootstrap-based interface with gradient backgrounds
- **Real-time Analysis**: Instant fraud detection with ML model integration
- **Interactive Visualizations**: Risk gauge charts and dynamic alerts
- **Sample Data**: Pre-filled legitimate and suspicious transaction examples
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **API Integration**: RESTful API endpoints for external integration
- **Comprehensive Results**: Risk factors, validation checks, and security recommendations

## 🚀 Quick Start

### Option 1: Using the Deployment Script (Recommended)
```bash
python deploy_web.py
```


### Option 2: Manual Setup
1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

2. Run the web application:
  ```bash
  python web_app.py
  ```

3. Open your browser and go to: `http://localhost:5000`

## 📱 Usage

1. **Enter Transaction Details**: Fill in the transaction form with sender/recipient information
2. **Load Sample Data**: Use the "Load Sample Payment" or "Load Suspicious" buttons for testing
3. **Analyze**: Click "Analyze Transaction" to get fraud risk assessment
4. **Review Results**: View risk level, probability, factors, and recommendations

## 🎯 Risk Assessment Levels

- **🟢 Low Risk (0-30%)**: Transaction approved, safe to proceed
- **🟡 Medium Risk (30-70%)**: Additional verification recommended
- **🔴 High Risk (70-100%)**: Transaction blocked, manual review required

## 🔧 API Endpoints

### Analyze Transaction
- **URL**: `POST /api/analyze`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "step": 1,
    "type": "PAYMENT",
    "amount": 5000.00,
    "oldbalanceOrg": 10000.00,
    "newbalanceOrg": 5000.00,
    "oldbalanceDest": 2000.00,
    "newbalanceDest": 7000.00,
    "sender": "John Doe",
    "recipient": "Jane Smith"
  }
  ```

### Get Sample Data
- **URL**: `GET /api/sample-data`
- **Response**: Returns sample legitimate and suspicious transaction data

## 🔒 Security Features

- **Risk Factor Analysis**: Identifies suspicious patterns
- **Transaction Validation**: Checks balance consistency and amount reasonableness
- **Contextual Recommendations**: Provides specific security actions based on risk level
- **Real-time Monitoring**: Instant analysis without data storage

## 🌐 Deployment Options

### Local Development
- Run with `python web_app.py`
- Access at `http://localhost:5000`

### Production Deployment
- Use with gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 web_app:app`
- Deploy to cloud platforms (Heroku, AWS, Google Cloud, etc.)
- Set up reverse proxy with nginx for production

## 📊 Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome
- **Charts**: Chart.js
- **ML Model**: Scikit-learn



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - feel free to use and modify as needed.

---

**🛡️ Built with security in mind - Professional fraud detection at your fingertips!**
