import subprocess
import sys
import os
import tkinter as tk
from tkinter import filedialog
tkroot = tk.Tk()
tkroot.withdraw()

def install():
    root = os.path.join(os.path.dirname(__file__)).replace("\\", "/") + "/"
    #directory = filedialog.askdirectory() + "/"
    #executable = sys.executable
    #with open(directory + "scNodes.bat", "w") as f:
    #    f.write('@echo off\n"' + str(executable) + '" "' + root + '__main__.py' + '"\npause')

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", root+"wheels/pyGpufit-1.2.0-py2.py3-none-any.whl", "pyGpufit"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", root+"wheels/dill-0.3.6-py3-none-any.whl", "dill"])
    try:
        print("Trying to install pyimgui-wheels. This may take a couple of minutes.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", root+"wheels/pyimgui-wheels-2.0.1.tar.gz"])
    except Exception as e:
        print("Installation of pyimgui-wheels failed - try running the following command:\npip install pyimgui-wheels\nand if succesful:\npython -m scNodes")

if __name__=="__main__":
    install()
