## Imports and constants
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(20) 
np.random.seed(20)


MAX_SEQ_LEN = 200
BATCH_SIZE = 10


## Simple model, without embedding layer
class SimpleLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_lstm_layers=1, cell_dropout=0.0, device=DEVICE):
        ## vocab_size = no. of unique words in reviews
        ## embedding_dim = size of embeddings / vectors
        ## hidden_dim = dimension of LSTM output
        ## num_lstm_layers = no. of LSTM layers
        ## cell_dropout = dropout applied between LSTM layers

        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_dim = hidden_dim
        self.device = device

        ## Model layers
            ## LSTM (for thought vector)
            ## Linear layer (for logit score)
            ## Activation (for P of +ve sentiment)

        self.model = nn.ModuleDict({
            'lstm': nn.LSTM(
                input_size=embedding_dim, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_lstm_layers, 
                batch_first=True, 
                dropout=cell_dropout,
                device=self.device
            ),
            'linear1': nn.Linear(
                in_features=self.hidden_dim, 
                out_features=3, ## 3 units for predicting 3 sentiments
                device=self.device
            ),
            'sigmoid': nn.Sigmoid()
        })

    
    def forward(self, x):
        ## Input is a (batch_size, sequence_length, feature_size) tensor
        hidden = self.init_hidden(len(x))
        x.to(self.device)

        ## LSTM outputs
            ## h_t = Tensor of shape (batch_size, sequence_length, direction*hidden_size) representing hidden state at each t
            ## h_n = Hidden state at last time step
            ## c_n = Cell state at last time step
        _, (h_n, _) = self.model['lstm'](x)
        # print(f'LSTM hidden states: {h_t.shape}')
        # print(f'LSTM final state: {h_n.shape}')

        output = self.model['linear1'](h_n[-1])
        # print(f'Linear output: {output.shape}')

        output = self.model['sigmoid'](output)
        # print(f'Sigmoid output: {output.shape}')

        return output.to(self.device), h_n[-1].to(self.device) ## return output of forward pass as well as thought vector


    ## Initialize initial cell and hidden states
    def init_hidden(self, batch_size):
        ret_tensor = torch.zeros(size=(self.num_lstm_layers, batch_size, self.hidden_dim))
        ret_tensor.to(self.device)

        return (ret_tensor, ret_tensor)


## Model with embedding layer 
class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_lstm_layers=1, cell_dropout=0.0, device=DEVICE):
        ## vocab_size = no. of unique words in reviews
        ## embedding_dim = size of embeddings / vectors
        ## hidden_dim = dimension of LSTM output
        ## num_lstm_layers = no. of LSTM layers
        ## cell_dropout = dropout applied between LSTM layers

        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_dim = hidden_dim
        self.device = device

        ## Model layers
        self.model = nn.ModuleDict({
            'embedding': torch.nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=embedding_dim, 
                device=self.device
            ),
            'lstm': nn.LSTM(
                input_size=embedding_dim, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_lstm_layers, 
                batch_first=True, 
                dropout=cell_dropout),
            'linear1': nn.Linear(
                in_features=self.hidden_dim, 
                out_features=3 ## 3 units for predicting 3 sentiments
            ),
            'sigmoid': nn.Sigmoid()
        })

    
    def forward(self, x):
        ## Input is a ...
        output = self.model['embedding'](x.long())

        h_0, c_0 = self.init_hidden(len(x))
        _, (h_n, _) = self.model['lstm'](output, (h_0, c_0)) ## LSTM outputs: h_t, h_n, c_n
        output = self.model['linear1'](h_n[-1])
        output = self.model['sigmoid'](output)
        output.to(self.device)

        return output, h_n[-1] ## return output of forward pass as well as thought vector
    

    ## Initialize initial cell and hidden states
    def init_hidden(self, batch_size):
        ret_tensor = torch.zeros(size=(self.num_lstm_layers, batch_size, self.hidden_dim))
        ret_tensor.to(self.device)

        return (ret_tensor, ret_tensor)