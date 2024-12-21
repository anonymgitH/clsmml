#!/bin/bash

# Installation Script for CLS-MML

echo "🚀 Starting Installation of CLS-MML..."

# Step 1: Create a virtual environment (recommended)
echo "📦 Creating a virtual environment..."
python -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip and install dependencies
echo "📥 Installing required dependencies..."
pip install --upgrade pip
pip install numpy scipy


# Step 4: Create an example script
echo "▶️ Creating an example script..."
cat <<EOF > example.py
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

# Importing functions from the code

# Example Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 2.9, 4.2, 4.8, 5.9])

# Run causal direction analysis
direction, confidence = causal_direction(x, y)
print(f"Causal Direction: {direction}, Confidence: {confidence}")
EOF

# Step 5: Run the example
echo "▶️ Running the example script..."
python example.py

