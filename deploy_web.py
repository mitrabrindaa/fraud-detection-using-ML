#!/usr/bin/env python3
"""
Deployment script for the Fraud Detection Web Application
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def run_web_app():
    """Run the Flask web application"""
    print("🚀 Starting Fraud Detection Web Application...")
    print("🌐 Features:")
    print("   • Professional Bootstrap UI")
    print("   • Real-time fraud analysis")
    print("   • Interactive risk gauge")
    print("   • Comprehensive risk assessment")
    print("   • Sample data loading")
    print("   • Responsive design")
    print("   • API endpoints for integration")
    print("")
    print("📱 Web app will be available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/analyze")
    print("")
    
    try:
        # Set environment variables for Flask
        os.environ['FLASK_APP'] = 'web_app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # Run Flask app
        subprocess.run([sys.executable, "web_app.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web app: {e}")
    except KeyboardInterrupt:
        print("\n👋 Web application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    print("🛡️  Fraud Detection Web Application Deployment")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists('models/model.pkl'):
        print("❌ Model file not found. Please run this script from the project root directory")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n" + "=" * 55)
    print("🎯 Ready to launch!")
    input("Press Enter to start the web application...")
    
    # Run the web application
    run_web_app()

if __name__ == "__main__":
    main()
