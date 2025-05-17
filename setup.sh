#!/bin/bash

# Exit immediately if any command fails
set -e

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "Virtual environment created and requirements installed."

# Optional: login to wandb (user can do this manually later if needed)
echo "If using wandb, login using: wandb login"
