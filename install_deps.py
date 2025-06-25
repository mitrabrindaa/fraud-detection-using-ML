import os
import subprocess
import sys

# Set thread limits FIRST
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

REQUIREMENTS = [
    "numpy==1.23.5",
    "scipy==1.9.3",
    "scikit-learn==1.2.2", 
    "xgboost==1.7.5",
    "pandas==1.5.3"
]

def install():
    for package in REQUIREMENTS:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--no-deps",
            "--ignore-installed",
            package
        ], check=True)

if __name__ == "__main__":
    install()