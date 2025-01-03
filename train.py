import torch 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from datetime import datetime

from utils.config import Config
from models.base import ImageCaptioningModel
from models.attention_model import ImageCaptioningWithAttention
from data.data_set import DataSet
from text.tokenizer import Tokenizer
import wandb




class EarlyStopping:
    def __init__(self, model, path, patience=10, delta=0):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class Trainer:
    def __init__(self, image_captioning_model, config):
        self.config = config
        self.image_captioning_model = image_captioning_model.to(self.config.device)
        
        self.optimizer = Adam(self.image_captioning_model.parameters(), lr=self.config.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.criterion = CrossEntropyLoss(ignore_index=self.config.padding_idx)
        self.early_stopping = EarlyStopping(
            self.image_captioning_model, 
            patience=10, 
            delta=0.01, 
            path=f"models/checkpoints/{self.config.model_name}/best_model.pth")
                
        
    def train_one_epoch(self, data_loader):
        self.image_captioning_model.train()
        total_loss = 0
        for images, captions in data_loader:
            images = images.to(self.config.device)
            captions = captions.to(self.config.device)
            
            self.optimizer.zero_grad()
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = self.image_captioning_model(images, inputs)
            if self.config.model_type == "attention":
                outputs = outputs.reshape(-1, self.config.vocab_size)
                loss = self.criterion(outputs, targets.reshape(-1))
            else:
                outputs = outputs[:, :-1, :].reshape(-1, outputs.shape[-1])
                loss = self.criterion(outputs, targets.reshape(-1))
            
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            
        return total_loss / len(data_loader)
    
    def validate(self, data_loader):
        self.image_captioning_model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, captions in data_loader:
                images = images.to(self.config.device)
                captions = captions.to(self.config.device)  
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                outputs = self.image_captioning_model(images, inputs)
                
                if self.config.model_type == "attention":
                    outputs = outputs.reshape(-1, self.config.vocab_size)
                    loss = self.criterion(outputs, targets.reshape(-1))
                else:
                    outputs = outputs[:, :-1, :].reshape(-1, outputs.shape[-1])
                    loss = self.criterion(outputs, targets.reshape(-1))

                total_loss += loss.item()
                
        return total_loss / len(data_loader)
                
            
    def train(self, train_loader, val_loader):
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self.early_stopping(val_loss)
            self.scheduler.step(val_loss)
            print('Learning Rate: ', self.scheduler.get_last_lr())
            if epoch % 10 == 1:
                torch.save(self.image_captioning_model.state_dict(), f"models/checkpoints/{self.config.model_name}/epoch_{epoch}.pth")
            
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, "learning_rate": self.scheduler.get_last_lr()[0]})


if __name__ == "__main__":

    os.makedirs(os.path.join("models/checkpoints"), exist_ok=True)
    experiment_id = 12
    config = Config(experiment_id=experiment_id)
    os.makedirs(os.path.join("models/checkpoints", config.model_name), exist_ok=True)

    print(f"Experiment {experiment_id}, name {config.model_name}")
    
    tokenizer = Tokenizer()
    tokenizer.load_dicts(config.path + "/dicts.pkl")
    
    if config.model_type == "static":
        
        image_captioning_model = ImageCaptioningModel(
            config.embedding_dim, 
            config.hidden_dim, 
            config.vocab_size,
            config.num_layers,
            config.dropout_rate
        ).to(config.device)
        
    elif config.model_type == "attention":
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        attention_dim = config.attention_dim
        decoder_dim = config.decoder_dim
        encoder_dim = config.encoder_dim

        image_captioning_model = ImageCaptioningWithAttention(
            embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim
            ).to(config.device)
        
    run = wandb.init(
        # Set the project where this run will be logged
        project="imageCaptioning",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config.learning_rate,
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "hidden_dim": config.hidden_dim,
            "attention_dim": config.attention_dim,
            "decoder_dim": config.decoder_dim,
            "encoder_dim": config.encoder_dim,
            "embed_size": config.embed_size,
            "model_name": config.model_name,
            "model_type": config.model_type,
        },
    )
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    
    train_set = DataSet(config.path, transform, tokenizer, data_type="train")
    val_set = DataSet(config.path, transform, tokenizer, data_type="val")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)
    
    trainer = Trainer(image_captioning_model, config)
    trainer.train(train_loader, val_loader)
        
        
