import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from config import Config
from models.models import ImageCaptioningModel
from models.attention_model import ImageCaptioningWithAttention
from data.data_set import DataSet
from text.tokenizer import Tokenizer

class CaptionPredictor:
    def __init__(self, model, transform, tokenizer, config):
        self.model = model
        self.transform = transform
        self.tokenizer = tokenizer
        self.config = config

        self.model.eval()

    def predict_single_image(self, image):
        with torch.no_grad():
            features = self.model.encoder(image.unsqueeze(0).to(self.config.device))
            caption = [self.tokenizer.word_to_idx['<start>']]
            
            for _ in range(self.config.max_length):
                inputs = torch.tensor([caption]).to(self.config.device)
                outputs = self.model.decoder(features, inputs)
                temperature = 0.8  
                probs = torch.softmax(outputs[:, -1, :] / temperature, dim=-1)
                word = torch.multinomial(probs, 1).item()
                caption.append(word)
                if word == self.tokenizer.word_to_idx['<end>']:
                    break

        predicted_text = ' '.join([self.tokenizer.idx_to_word[idx] for idx in caption])
        return predicted_text
    
    def predict_multiple_images(self, images):
        predicted_captions = []
        for image in images:
            predicted_text = self.predict_single_image(image)
            predicted_captions.append(predicted_text)
        return predicted_captions
    
    def save_predictions(self, model_path, image_paths, true_captions, predicted_captions):
        results_df = pd.DataFrame({
            'True Caption': true_captions,
            'Predicted Caption': predicted_captions
        })
        results_df['Image Path'] = image_paths
        results_df.to_csv(f'{model_path}/predictions.csv', index=False)


if __name__ == "__main__":
    
    MODEL_PATH = "models/checkpoints/new_runs/run_11"
    
    config = Config(experiment_id=11)
    #model = ImageCaptioningModel(config.embedding_dim, config.hidden_dim, config.vocab_size, config.num_layers, config.dropout_rate).to(config.device)
    model = ImageCaptioningWithAttention(config.embed_size, config.attention_dim, config.decoder_dim, config.vocab_size, config.encoder_dim).to(config.device)
    model.load_state_dict(torch.load(f"{MODEL_PATH}/best_model.pth", map_location=config.device, weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = Tokenizer()
    tokenizer.load_dicts(config.path + "/dicts.pkl")
    
    test_set = DataSet(config.path, transform, tokenizer, data_type="test")
    test_loader = DataLoader(test_set, batch_size=4)
    
    predictor = CaptionPredictor(model, transform, tokenizer, config)
    predicted_captions = []
    true_captions = []
    image_names = []
    for i, (images, captions) in tqdm(enumerate(test_loader)):
        predicted_captions.extend(predictor.predict_multiple_images(images))
        true_captions.extend([tokenizer.decode_caption(caption) for caption in captions])
        image_names.extend(images)
    
    predictor.save_predictions(MODEL_PATH, image_names, true_captions, predicted_captions)