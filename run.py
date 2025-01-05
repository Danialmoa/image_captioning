from data.data_set import DataSet
from inference.caption_predictor import CaptionPredictor
from utils.config import Config
import torch
from models.attention_model import ImageCaptioningWithAttention
from text.tokenizer import Tokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def run():
    
    model_path = "models/final_model/best_model.pth"
    config = Config(experiment_id='best')
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = ImageCaptioningWithAttention(
        config.embed_size, 
        config.attention_dim, 
        config.decoder_dim, 
        config.vocab_size, 
        config.encoder_dim).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = Tokenizer()
    tokenizer.load_dicts(config.path + "/dicts.pkl")
    
    predictor = CaptionPredictor(model, transform, tokenizer, config, device)
    
    sample_set = DataSet(config.path, transform, tokenizer, data_type="test")
    print(sample_set[0])

    sample_loader = DataLoader(sample_set, batch_size=4, shuffle=False)
    
    for i, (images, captions, image_names) in tqdm(enumerate(sample_loader), total=len(sample_loader), desc="Predicting captions"):
        predicted_captions = predictor.predict_multiple_images(images)
        true_captions = [tokenizer.decode_caption(caption) for caption in captions]
        print(predicted_captions)
        print(true_captions)
        break

if __name__ == "__main__":
    run()
