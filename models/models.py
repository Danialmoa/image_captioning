import torch
import torch.nn as nn
from torchvision import models
from model2vec import StaticModel


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        
        if embedding_dim != 2048:
            self.fc = nn.Linear(2048, embedding_dim)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.data.fill_(0)
            
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
    def forward(self, images):
        with torch.no_grad():
            features = self.feature_extractor(images)
        features = features.view(features.size(0), -1) 
        if self.embedding_dim != 2048:
            features = self.fc(features)
        return features
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.fc(hiddens)
        return outputs
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout_rate=0.5):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, vocab_size, num_layers, dropout_rate)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs



if __name__ == "__main__":

    model = ImageCaptioningModel(
        embedding_dim=256, hidden_dim=256, vocab_size=8784, num_layers=1)
    print(model)
    