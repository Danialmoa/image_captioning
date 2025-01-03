#!/bin/bash

# Exit on error
set -e

# Data path
RAW_DATA_DIR="data/raw"
PROCESSED_DATA_DIR="data/processed"

echo "🚀 Starting data preparation..."

# Activate the virtual environment
source .venv/bin/activate

# Install unzip
if ! command -v unzip &> /dev/null; then
    echo "📥 Installing unzip..."
    sudo apt-get install unzip
fi

# Download dataset 
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "📥 Downloading dataset..."
    mkdir -p $RAW_DATA_DIR
    
    # Download the dataset
    read -p "You can select two datasets, flickr8k or flickr30k, which one do you want to download? (1- flickr8k/2- flickr30k): " dataset
    if [ "$dataset" == "1" ]; then
        curl -L -o $RAW_DATA_DIR/flickr-image-dataset.zip\
            https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
        
        echo "📂 Unzipping dataset..."
        unzip -q $RAW_DATA_DIR/flickr-image-dataset.zip -d $RAW_DATA_DIR
        mv $RAW_DATA_DIR/Images $RAW_DATA_DIR/images

    elif [ "$dataset" == "2" ]; then
        curl -L -o $RAW_DATA_DIR/flickr-image-dataset.zip\
            https://www.kaggle.com/api/v1/datasets/download/hsankesara/flickr-image-dataset
        
        echo "📂 Unzipping dataset..."
        unzip -q $RAW_DATA_DIR/flickr-image-dataset.zip -d $RAW_DATA_DIR
        mkdir -p $RAW_DATA_DIR/images
        mv $RAW_DATA_DIR/flickr30k_images/flickr30k_images/*.jpg $RAW_DATA_DIR/images/
        mv $RAW_DATA_DIR/flickr30k_images/results.csv $RAW_DATA_DIR/captions.txt
    else
        echo "Invalid dataset selection. Please select either flickr8k or flickr30k."
        exit 1
    fi
    

    # Remove the zip file
    echo "📂 Removing zip file..."
    rm $RAW_DATA_DIR/flickr-image-dataset.zip
fi

# Check if processed data exists
if [ -d "$PROCESSED_DATA_DIR" ]; then
    echo "⚠️  Processed dataset already exists!"
    while true; do
        read -p "Do you want to delete existing processed data and create a new one? (y/n): " yn
        case $yn in
            [Yy]* )
                echo "🗑️  Removing existing processed dataset..."
                rm -rf "$PROCESSED_DATA_DIR"
                break
                ;;
            [Nn]* )
                echo "⏭️  Skipping preprocessing, using existing dataset..."
                exit 0
                ;;
            * )
                echo "Please answer y or n."
                ;;
        esac
    done
fi

if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    # Create processed data directory and run preprocessing
    mkdir -p $PROCESSED_DATA_DIR
    echo "🔑 Preprocessing dataset..."
    python data/pre_processing.py
    echo "🔑 Tokenizing dataset..."
    python text/tokenizer.py
fi

echo "✅ Data preparation completed successfully!"