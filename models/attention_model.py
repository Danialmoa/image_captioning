import torch 
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, embed_size=256):
        super(Encoder, self).__init__()
        #resnet = models.resnet50(weights='IMAGENET1K_V1') # Change from v10
        resnet = models.resnet101(weights='IMAGENET1K_V2') 
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1) 
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2048, embed_size, kernel_size=1),
        #     nn.BatchNorm2d(embed_size),
        #     nn.ReLU(inplace=True) 
        # ) # EXP 11

    def forward(self, images):
        features = self.feature_extractor(images) 
        features = self.conv(features) 
        return features
    

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, encoder_features, decoder_hidden):
        batch_size, embed_size, H, W = encoder_features.size()
        encoder_features = encoder_features.reshape(batch_size, embed_size, -1).transpose(1, 2)
        
        encoder_att = self.encoder_att(encoder_features)
        decoder_att = self.decoder_att(decoder_hidden)
        decoder_att = decoder_att.unsqueeze(1)
        att = self.full_att(self.relu(encoder_att + decoder_att))
        att = att.squeeze(2)
        alpha = self.softmax(att)

        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1) 
        return context, alpha
        
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim=256, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_features, captions):
        batch_size = encoder_features.size(0)
        seq_length = captions.size(1)
        vocab_size = self.fc.out_features

        h, c = self.init_hidden_state(batch_size)

        embeddings = self.embedding(captions)

        outputs = torch.zeros(batch_size, seq_length, vocab_size).to(encoder_features.device)
        for t in range(seq_length):
            context, _ = self.attention(encoder_features, h) 
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1) 
            h, c = self.lstm(lstm_input, (h, c)) 
            output = self.fc(self.dropout(h))
            outputs[:, t, :] = output

        return outputs

    def init_hidden_state(self, batch_size):
        return (torch.zeros(batch_size, self.lstm.hidden_size).to(next(self.parameters()).device),
                torch.zeros(batch_size, self.lstm.hidden_size).to(next(self.parameters()).device))
        
        
class ImageCaptioningWithAttention(nn.Module):
    def __init__(self, embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim=256):
        super(ImageCaptioningWithAttention, self).__init__()
        self.encoder = Encoder(embed_size=encoder_dim)
        self.decoder = DecoderWithAttention(embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim)

    def forward(self, images, captions):
        encoder_features = self.encoder(images) 
        outputs = self.decoder(encoder_features, captions)
        return outputs
        
        
if __name__ == "__main__":
    import config as config
    from data.data_set import DataSet
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from text.tokenizer import Tokenizer
    from torch.nn import CrossEntropyLoss
    
    
    config = config.Config(7)
    device = config.device
    vocab_size = config.vocab_size
    embed_size = config.embed_size
    attention_dim = config.attention_dim
    decoder_dim = config.decoder_dim
    encoder_dim = config.encoder_dim

    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
    tokenizer = Tokenizer()
    tokenizer.load_dicts(config.path + "/dicts.pkl")
    # Sample Inputs
    
    data_set = DataSet(config.path, transform, tokenizer, data_type="train", sample_size=10)
    data_loader = DataLoader(data_set, batch_size=2, shuffle=True)
    
    images, captions = next(iter(data_loader))
    images = images.to(device)
    captions = captions.to(device)
    
    model = ImageCaptioningWithAttention(embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim)
    model.to(device)

    outputs = model(images, captions[:, :-1])
    targets = captions[:, 1:]
    print("Outputs shape:", outputs.shape, "Targets shape:", targets.shape, "Inputs shape:", images.shape, "Captions shape:", captions.shape)
    print(outputs.reshape(-1, vocab_size), targets.reshape(-1))
    
    criterion = CrossEntropyLoss()
    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
    print("Loss:", loss.item())
    
    loss.backward()
    
    #torch.onnx.export(model, (images, captions), "models/attention_model.onnx", opset_version=11)
    