# Tomatoes
opencv python 

# Installation Guide

This guide provides instructions on setting up Python3 along with OpenCV and NumPy on the Windows Subsystem for Linux (WSL). Linux users can follow along from the "Linux Users" section.

## Prerequisites

- Windows 10 or later with WSL enabled.
- Access to a Linux distribution (e.g., Ubuntu) via WSL. [Learn how to set up WSL](https://docs.microsoft.com/en-us/windows/wsl/install).

## Linux Users

From this point, Linux users outside WSL can follow the guide identically. Ensure you're operating within a Python virtual environment or a Conda environment when installing packages to manage dependencies effectively.

## Step 1: Setting up Python3

Ensure that Python3 is installed on your system. Most Linux distributions come with Python pre-installed. You can check your Python version by running:

```bash
    python3 --version
```   
## If Python3 is not installed, you can install it using your distribution's package manager. For Ubuntu or Debian-based distributions:

```bash
    sudo apt update
    sudo apt install python3 python3-pip
```

## Step 2: Using Virtual Environments
Option A: Virtual Environment (venv)

It's recommended to use a virtual environment for Python projects to manage dependencies. To create a virtual environment, navigate to your project directory and run:

```bash
    python3 -m venv myenv
```

To activate the virtual environment:

```bash
    source myenv/bin/activate
```
Option B: Conda Environment

If you prefer using Conda, first ensure that Anaconda or Miniconda is installed. Create a new Conda environment by running:
```bash    
    conda create --name myenv python=3.12.1
```
To activate the Conda environment:
```bash
    conda activate myenv
```
## Step 3: Installing OpenCV and NumPy
With your virtual environment activated, install OpenCV and NumPy using pip:
```bash
    pip install numpy opencv-python
```

## Step 4: Verifying the Installation
To verify that OpenCV and NumPy have been installed correctly, you can run the following commands in your Python shell:
```bash
    import cv2
    import numpy as np

    print(cv2.__version__)
    print(np.__version__)
```
If the versions of OpenCV and NumPy are displayed without any errors, the installation was successful.

## Conclusion

You now have a working setup of Python3 with OpenCV and NumPy on WSL (or Linux). Remember to activate your virtual or Conda environment whenever you're working on your project to keep your dependencies organized and avoid conflicts.
