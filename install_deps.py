# install_deps.py
import subprocess
import sys

REQUIREMENTS = [
    "numpy==1.26.4",
    "scipy==1.13.1",
    "scikit-learn==1.3.2",
    "xgboost==2.0.3",
    "pandas==2.1.4"
]

def install():
    for package in REQUIREMENTS:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-deps",  # Critical flag
            "--ignore-installed",
            package
        ])

if __name__ == "__main__":
    install()