from pathlib import Path
import string
import re
import pickle
import torch
from utils.config import Config

import numpy as np

class Tokenizer:    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.config = Config()
        self.max_seq_length = 0
        
    def save_dicts(self, path: str):
        """
        Saves the dictionaries to the given path.
        """
        with Path(path).open("wb") as file:
            pickle.dump((self.word_to_idx, self.idx_to_word), file)
        
    def load_dicts(self, path: str):
        """
        Loads the dictionaries from the given path.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist")
        with Path(path).open("rb") as file:
            (self.word_to_idx, self.idx_to_word) = pickle.load(file)
                 
    @staticmethod            
    def word_tokenization(text: str):
        """
        Simple tokenizer that splits by spaces.
        """
        return text.split()
    
    def create_vocab(self, indexs):
        """
        Creates a vocabulary from the captions.
        """
        index = 0
        for line in indexs:
            _, _, caption = line.split(",")
            tokens = self.word_tokenization(caption)
                
            if len(tokens) > self.max_seq_length:
                self.max_seq_length = len(tokens)
                
            for token in tokens:
                if token not in self.word_to_idx:
                    self.word_to_idx[token] = index
                    self.idx_to_word[index] = token
                    index += 1

        # add padding, OOV, start and end tokens to the vocabulary and update the dictionaries
        extra_caracters = ["<pad>", "<unk>", "<start>", "<end>"]
        for i in range(4):
            self.word_to_idx[extra_caracters[i]] = index + i
            self.idx_to_word[index + i] = extra_caracters[i]
            index += 1

        self.max_seq_length += 2 # add the 2 extra characters to the max sequence length (Start and End tokens)

        print('vocab size', len(self.word_to_idx))
        print('max seq length', self.max_seq_length)
        print('padding idx', self.word_to_idx["<pad>"])
        
        
    # def use_model2vec(self, text, max_length=config.max_length):
    #     """
    #     Uses the model2vec to tokenize the text.
    #     """
    #     model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    #     text = ["<start>"] + self.word_tokenization(text) + ["<end>"] 
    #     text = text + ["<pad>"] * (max_length - len(text))
    #     embeddings = np.zeros((len(text), model.dim))
    #     for i, each_word in enumerate(text):
    #         embeddings[i] = model.encode(each_word)
    #     embeddings = torch.tensor(embeddings)
        
    #     return embeddings.float()
    
    
    def create_seq_tensor(self, text):
        """
        Creates a tensor of sequences from the tokenized captions.
        """
        if not self.word_to_idx :
            raise ValueError("Word2idx dictionary is not initialized")
        
        seq_tensor = [self.word_to_idx["<start>"]]
        tokens = self.word_tokenization(text)
        for token in tokens:
            if token in self.word_to_idx:
                seq_tensor.append(self.word_to_idx[token])
            else:
                seq_tensor.append(self.word_to_idx["<unk>"])
        
        seq_tensor.append(self.word_to_idx["<end>"])
        
        if len(seq_tensor) < self.config.max_length:
            seq_tensor.extend([self.word_to_idx["<pad>"]] * (self.config.max_length - len(seq_tensor)))
        
        return torch.tensor(seq_tensor, dtype=torch.long)
    
    def decode_caption(self, seq_tensor):
        """
        Decodes a sequence tensor to a caption.
        """
        return ' '.join([self.idx_to_word[idx.item()] for idx in seq_tensor])
        

if __name__ == "__main__":
    text_manager = Tokenizer()
    
    train_indexs = open("data/processed/train/index.txt", "r").readlines()
    val_indexs = open("data/processed/val/index.txt", "r").readlines()
    test_indexs = open("data/processed/test/index.txt", "r").readlines()
    
    indexs = train_indexs + val_indexs + test_indexs
    
    #text_manager.create_vocab(indexs)
    #text_manager.save_dicts("data/processed/dicts.pkl")
    
    text_manager.load_dicts("data/processed/dicts.pkl")

    #create the sequence tensor
    text = "a two child in a pink dress is climbing up a set of stairs in an entry way ."
    new_text = ""
    seq_tensor = text_manager.create_seq_tensor(text)
    print(seq_tensor)
    print(seq_tensor.shape)
    for i in range(len(seq_tensor)):
        new_text += text_manager.idx_to_word[seq_tensor[i].item()] + ' '
    print(new_text)
    print(len(text_manager.word_to_idx))
    