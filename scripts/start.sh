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

# Login to wandb (will prompt for API key)
echo "🔑 Logging in to Weights & Biases..."
while true; do
    echo "🔑 Please enter your Weights & Biases API key:"
    read WANDB_API_KEY 
    if [ -z "$WANDB_API_KEY" ]; then
        echo "API key cannot be empty. Please try again."
        continue
    fi
    echo "Attempting to login to Weights & Biases..."
    export WANDB_API_KEY=$WANDB_API_KEY
    if wandb login; then
        echo "✅ Successfully logged in to Weights & Biases!"
        break
    else
        echo "❌ Invalid API key. Please try again."
    fi
done

echo "✅ Setup completed successfully!"