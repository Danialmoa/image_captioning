import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from text.tokenizer import Tokenizer
from utils.config import Config

class DataSet(Dataset):
    """
    DataSet class for the image captioning task.
    """
    def __init__(self, path, transform, tokenizer, data_type="train", sample_size=None):
        self.path = path + "/" + data_type
        self.caption_lists = self.pre_load_text(self.path + "/index.txt")
        random.shuffle(self.caption_lists)
        if sample_size:
            self.caption_lists = self.caption_lists[:sample_size]
        
        self.tokenizer = tokenizer 
        self.transform = transform         
        
    def __len__(self):
        """
        Returns the number of captions in the dataset.
        """
        return len(self.caption_lists)
    
    def pre_load_text(self, path):
        """
        Pre-loads the text from the given path.
        """
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines
        
        
    def __getitem__(self, idx):
        """
        Returns the image and target at the given index.
        """
        index, image_name , caption = self.caption_lists[idx].split(",")
        image_path = os.path.join(self.path, "images", image_name)
        
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        text_tensor = self.tokenizer.create_seq_tensor(caption)
        caption_tensor = torch.as_tensor(text_tensor, dtype=torch.long)
        
        return image, caption_tensor

    

if __name__ == "__main__":
    config = Config()
    path = config.path
    
    tokenizer = Tokenizer()
    tokenizer.load_dicts(path + "/dicts.pkl")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Resize the image to 256x256
        transforms.ToTensor(), # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]) # Normalize the image base on ResNet50 
    ])
    data_set = DataSet(path, transform, tokenizer, sample_size=10)
    data_loader = DataLoader(data_set, batch_size=16, shuffle=True)
    print(data_set[0][1].shape)