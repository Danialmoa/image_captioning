import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from utils.config import Config
from models.base import ImageCaptioningModel
from models.attention_model import ImageCaptioningWithAttention
from data.data_set import DataSet
from text.tokenizer import Tokenizer

class CaptionPredictor:
    def __init__(self, model, transform, tokenizer, config, device, temperature=0.8):
        """
        Initialize the caption predictor
        args:
            model: the model to use for predicting the caption
            transform: the transform to use for the image
            tokenizer: the tokenizer to use for the caption
            config: the config to use for the model
            device: the device to use for the model
            temperature: the temperature to use for the model 
        """
        self.model = model
        self.transform = transform
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.model.eval()
        self.temperature = temperature

    def predict_single_image(self, image):
        """
        Predict the caption for a single image
        args:
            image: the image to predict the caption for
        returns:
            the predicted caption
        """
        with torch.no_grad():
            features = self.model.encoder(image.unsqueeze(0).to(self.device))
            caption = [self.tokenizer.word_to_idx['<start>']] # start token
            
            # start predicting the caption and it will stop when it reaches the end token
            for _ in range(self.config.max_length): 
                inputs = torch.tensor([caption]).to(self.device) 
                outputs = self.model.decoder(features, inputs)
                probs = torch.softmax(outputs[:, -1, :] / self.temperature, dim=-1) # get the probability of the next word
                word = torch.multinomial(probs, 1).item() 
                caption.append(word)
                if word == self.tokenizer.word_to_idx['<end>']:
                    break

        predicted_text = ' '.join([self.tokenizer.idx_to_word[idx] for idx in caption])
        return predicted_text
    
    def predict_multiple_images(self, images):
        """
        Predict the caption for multiple images
        args:
            images: the images to predict the caption for
        returns:
            the predicted captions
        """
        predicted_captions = []
        for image in images:
            predicted_text = self.predict_single_image(image)
            predicted_captions.append(predicted_text)
        return predicted_captions
    
    def save_predictions(self, model_path, image_names, true_captions, predicted_captions):
        """
        Save the predicted captions to a csv file
        args:
            model_path: the path to save the csv file
            image_names: the names of the images
            true_captions: the true captions
            predicted_captions: the predicted captions
        """
        results_df = pd.DataFrame({
            'TrueCaption': true_captions,
            'PredictedCaption': predicted_captions
        })
        results_df['ImageName'] = image_names
        results_df.to_csv(f'{model_path}/predictions_temp_{self.temperature}.csv', index=False)


if __name__ == "__main__":
    run_id = 14
    MODEL_PATH = f"models/checkpoints/run_{run_id}"
    TEMPERATURE = 0.8
    config = Config(experiment_id=run_id)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    #model = ImageCaptioningModel(config.embedding_dim, config.hidden_dim, config.vocab_size, config.num_layers, config.dropout_rate).to(config.device)
    model = ImageCaptioningWithAttention(config.embed_size, config.attention_dim, config.decoder_dim, config.vocab_size, config.encoder_dim).to(device)
    model.load_state_dict(torch.load(f"{MODEL_PATH}/best_model.pth", map_location=device, weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = Tokenizer()
    tokenizer.load_dicts(config.path + "/dicts.pkl")
    
    test_set = DataSet(config.path, transform, tokenizer, data_type="test")
    print(len(test_set))
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    
    predictor = CaptionPredictor(model, transform, tokenizer, config, device, temperature=TEMPERATURE)
    predicted_captions = []
    true_captions = []
    image_names_all = []
    for i, (images, captions, image_names) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Predicting captions"):
        predicted_captions.extend(predictor.predict_multiple_images(images))
        true_captions.extend([tokenizer.decode_caption(caption) for caption in captions])
        image_names_all.extend([this_image_name for this_image_name in image_names])

    predictor.save_predictions(MODEL_PATH, image_names_all, true_captions, predicted_captions)