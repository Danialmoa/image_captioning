from data.data_set import DataSet
from inference.caption_predictor import CaptionPredictor
from utils.config import Config
import torch
from models.attention_model import ImageCaptioningWithAttention
from inference.evaluation import analyze_results
from text.tokenizer import Tokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


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

    sample_loader = DataLoader(sample_set, batch_size=4, shuffle=False)
    
    final_df = pd.DataFrame(columns=["image_name", "true_caption", "predicted_caption"])
    
    predicted_captions = []
    true_captions = []
    image_names_all = []

    for i, (images, captions, image_names) in tqdm(enumerate(sample_loader), total=len(sample_loader), desc="Predicting captions"):
        predicted_captions.extend(predictor.predict_multiple_images(images))
        true_captions.extend([tokenizer.decode_caption(caption) for caption in captions])
        image_names_all.extend([this_image_name for this_image_name in image_names])
        
    final_df = analyze_results(true_captions, predicted_captions, image_names_all, "best", save=False)
    best_prediction_captions = final_df.loc[final_df.groupby('Image Name')['BLEU Score'].idxmax()]
    
    best_prediction_captions.sort_values(by="BLEU Score", ascending=False, inplace=True)
    for i in range(len(best_prediction_captions)):
        print("Image path: ", "data/sample_data/test/images/" + best_prediction_captions['Image Name'].iloc[i])
        print("True Caption: ", best_prediction_captions['True Caption'].iloc[i])
        print("Predicted Caption: ", best_prediction_captions['Predicted Caption'].iloc[i])
        print("BLEU Score: ", best_prediction_captions['BLEU Score'].iloc[i])
        print("-"*50)

if __name__ == "__main__":
    run()
