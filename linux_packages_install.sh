#!/bin/bash


# Activate the environment
source venv/bin/activate

# Install your packages
# Install the specific package (replace with your package names)
pip3 install torch torchvision
pip3 install optimum-quanto accelerate transformers -q
pip3 install install ninja wheel --upgrade
pip3 install matplotlib seaborn numpy pandas huggingface_hub scikit-learn -q

echo "Installation complete! Proceed to run QAT program."

