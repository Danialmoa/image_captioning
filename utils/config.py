class Config:
    def __init__(self, experiment_id=1):
        self.device = "cuda"
        self.num_epochs = 100
        self.vocab_size = 19779 #8785
        self.num_layers = 1 
        self.path = "data/processed"
        self.max_length = 80 #37
        self.padding_idx = 19772 #8778
        
        # In 1 to 3 experiment_id, I want to check the effec
        # of the embedding dimension and the hidden dimension on the performance of the model 
        # The feature Summarisation is different actually
        if experiment_id == 1:
            self.learning_rate = 0.0001
            self.batch_size = 32
            self.embedding_dim = 2048
            self.hidden_dim = 256
            self.model_name = "run_1"
            self.model_type = "static"
            self.dropout_rate = 0.5
        elif experiment_id == 2:
            self.learning_rate = 0.0001
            self.batch_size = 32
            self.embedding_dim = 1024
            self.hidden_dim = 256
            self.model_name = "run_2"
            self.model_type = "static"
            self.dropout_rate = 0.5
        elif experiment_id == 3:
            self.learning_rate = 0.0001
            self.batch_size = 32
            self.embedding_dim = 512
            self.hidden_dim = 256
            self.model_name = "run_3"
            self.model_type = "static"
            self.dropout_rate = 0.5
        elif experiment_id == 4:
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.embedding_dim = 512
            self.hidden_dim = 256
            self.model_name = "run_4" # -Deleted
            self.model_type = "static"
            self.dropout_rate = 0.5
            
        elif experiment_id == 5:
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.embedding_dim = 512
            self.hidden_dim = 256
            self.model_name = "run_5" # -> Run 4
            self.model_type = "static"
            self.dropout_rate = 0.3
            
        elif experiment_id == 6:
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.embedding_dim = 512
            self.hidden_dim = 256
            self.model_name = "run_6"
            self.model_type = "static"
            self.dropout_rate = 0.7
            
        
        elif experiment_id == 7:
            self.learning_rate = 0.0001
            self.batch_size = 32
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_7"
            self.model_type = "attention"
            
        elif experiment_id == 8:
            self.learning_rate = 0.0001
            self.batch_size = 64
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_8"
            self.model_type = "attention"
            
        elif experiment_id == 9:
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_9"
            self.model_type = "attention"
            
        elif experiment_id == 10:
            # FROM this example I've changed the model to resnet101
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_10"
            self.model_type = "attention"
            
            self.max_length = 37
            self.padding_idx = 8778
            self.vocab_size = 8785
        
        elif experiment_id == 11:
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_11"
            self.model_type = "attention"
            
            self.max_length = 37
            self.padding_idx = 8778
            self.vocab_size = 8785
            
        elif experiment_id == 12:
            # 12 is the same as 10 but with a different dataset
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_12"
            self.model_type = "attention"
            
        elif experiment_id == 13:
            # 12 is the same as 10 but with a different dataset
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_13"
            self.model_type = "attention"
            
        elif experiment_id == 14:
            # Add weight initialisation, reduce the scheduler factor, and add augmentation and color jitter
            self.learning_rate = 0.0001
            self.batch_size = 16
            self.hidden_dim = 256
            self.attention_dim = 256
            self.decoder_dim = 512
            self.encoder_dim = 256
            self.embed_size = 256
            self.model_name = "run_14"
            self.model_type = "attention"
            
            self.max_length = 37
            self.padding_idx = 8778
            self.vocab_size = 8785
            
            
config = Config()
