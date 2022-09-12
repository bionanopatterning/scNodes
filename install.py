import subprocess
import sys
import os

def install_modules():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", "pyGpufit-1.2.0-py2.py3-none-any.whl", "pyGpufit"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "req.txt"])

install_modules()