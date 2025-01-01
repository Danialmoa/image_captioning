#!/bin/bash

# Exit on error
set -e

# Data path
RAW_DATA_DIR="../data/raw"
PROCESSED_DATA_DIR="../data/processed"

echo "🚀 Starting data preparation..."

# Activate the virtual environment
source ../.venv/bin/activate

# Install unzip
echo "📥 Installing unzip..."
sudo apt-get install unzip

# Download dataset 
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "📥 Downloading dataset..."
    mkdir -p $RAW_DATA_DIR
    
    curl -L -o $RAW_DATA_DIR/flickr-image-dataset.zip\
        https://www.kaggle.com/api/v1/datasets/download/hsankesara/flickr-image-dataset 
    
    # Unzip the dataset
    echo "📂 Unzipping dataset..."
    unzip -q $RAW_DATA_DIR/flickr-image-dataset.zip -d $RAW_DATA_DIR
    mkdir -p $RAW_DATA_DIR/images
    mv $RAW_DATA_DIR/flickr30k_images/*.jpg $RAW_DATA_DIR/images/
    mv $RAW_DATA_DIR/flickr30k_images/results.csv $RAW_DATA_DIR/captions.txt

    # Remove the zip file
    echo "📂 Removing zip file..."
    rm $RAW_DATA_DIR/flickr-image-dataset.zip
fi

# Check if dataset exists
if [ -d "$PROCESSED_DATA_DIR" ]; then
    echo "⚠️  Dataset already exists!"
    while true; do
        read -p "Do you want to delete existing data and download again? (y/n): " yn
        case $yn in
            [Yy]* )
                echo "🗑️  Removing existing dataset..."
                rm -rf "$PROCESSED_DATA_DIR"
                mkdir -p $PROCESSED_DATA_DIR

                # Run the pre_processing.py script
                echo "🔑 Preprocessing dataset..."
                python data/pre_processing.py
                echo "🔑 Tokenizing dataset..."
                python text/tokenizer.py

                break
                ;;
            [Nn]* )
                echo "⏭️  Skipping download, using existing dataset..."
                exit 0
                ;;
            * )
                echo "Please answer y or n."
                ;;
        esac
    done
fi

echo "✅ Data preparation completed successfully!"