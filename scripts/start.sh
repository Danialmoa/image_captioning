#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting project setup..."

# Create and activate Python virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Install module
echo "📥 Installing module..."
pip install -e .

echo "✅ Setup completed successfully!"