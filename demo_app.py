#!/usr/bin/env python3
"""
Demo script for the Enhanced Fraud Detection App
This script demonstrates the new frontend features
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app with enhanced frontend"""
    try:
        # Change to the src directory
        os.chdir('src')
        
        print("🛡️  Starting Enhanced Fraud Detection System...")
        print("🚀 Features added:")
        print("   • Beautiful gradient background")
        print("   • Interactive risk gauge visualization")
        print("   • Feature importance charts")
        print("   • Quick-fill buttons for testing")
        print("   • Enhanced risk assessment with animations")
        print("   • Detailed transaction validation")
        print("   • Security recommendations")
        print("   • Professional styling with CSS")
        print("\n📱 Opening in your browser...")
        
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the app: {e}")
        print("💡 Make sure you're in the project directory and have all dependencies installed")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🎨 Enhanced Fraud Detection App Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('src/app.py'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    run_streamlit_app()
