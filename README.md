# Image Captioning Project
This project is a project for the course Nueral Networks, at the Univesity of Siena.

## Project Description
The project is about image captioning, which is a task of generating a natural language description of an image.

## Project Structure
The project is structured as follows:
- `data/`:  Dataset,  preprocessing scripts, processed data
    - `raw_data/`: contains the raw data
    - `processed/`: contains the processed data
    - `data_set/`: contains the data set class for loading the data
    - `pre_processing.py`: contains the code for preprocessing the data
- `inference/`: contains the code for prediction and evaluation
    - `caption_predictor.py`: contains the code for predicting the caption , use the checkpoint models to predict the caption base on test set
    - `evaluation.py`: contains the code for evaluating the model, use the prediction results to evalute the model base on BLEU score
- `models/`: contains the code for the models
    - `base_model.py`: contains the code for the base model - just a LSTM model and resnet50
    - `attention_model.py`: contains the code for the attention model - a LSTM model with attention mechanism
- `text/`: contains the code for the text processing and tokenizer
    - `tokenizer.py`: contains the code for the tokenizer, and make the vocabulary, and the word to index mapping
- `utils/`: contains just the config file
- `scripts/`: 
    - `start.sh`: contains the code for making environment and install the dependencies
    - `prepare_data.sh`: contains the code for downloading-preprocessing dataset
