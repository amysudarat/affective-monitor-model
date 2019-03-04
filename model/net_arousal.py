# -*- coding: utf-8 -*-

"""
Create LSTM model for predicting valence label

Model A: 1 Hidden Layer
-Unroll 100 time steps
    - Each step input size: 2 x 1
    - Total per unroll: 2 x 100
        -Feedforward Neural Network input size: 2 x 100
    - 1 Hidden layer
    -Output dimension : 4 
"""

import torch
import torch.nn as nn


class myLSTM_arousal(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim):
        super(myLSTM_arousal,self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building LSTM
        # batch_first = True causes input/output tensors to be of shape
        # (batch_dim,seq_dim,feature_dim) in other word put the index of batch
        # on the first column of tuple
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True)
        
        # Readout Layer (non-recurrent output layer)
        self.fc = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True)
        
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True)

        # call one time will do 100 time steps
        out, (hn,cn) = self.lstm(x,(h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 2
        # out[:, -1, :] --> 100, 2 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 2
        return out
