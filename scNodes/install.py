import subprocess
import sys
import os
import tkinter as tk
from tkinter import filedialog
tkroot = tk.Tk()
tkroot.withdraw()

def install():
    root = os.path.join(os.path.dirname(__file__)).replace("\\", "/") + "/"
    try:
        print("Trying to install pyGpufit, with wheel at:\n"+root+"wheels/pyGpufit-1.2.0-py2.py3-none-any.whl")
        subprocess.check_call([sys.executable, "-m", "pip", "install", root+"wheels/pyGpufit-1.2.0-py2.py3-none-any.whl"])
    except Exception as e:
        print("Installation of pyGpufit failed. Try installing the wheel manually - see https://github.com/gpufit")
        raise e

if __name__=="__main__":
    install()
