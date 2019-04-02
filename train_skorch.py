# -*- coding: utf-8 -*-

import torch
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from skorch import NeuralNetClassifier


""" Define Pytorch module to use"""
class myLSTM_arousal(nn.Module):
    def __init__(self,input_dim=1, hidden_dim=20, layer_dim=1, output_dim=5):
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
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True)
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, requires_grad = True)

        # call one time will do 100 time steps
        out, (hn,cn) = self.lstm(x,(h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 2
        # out[:, -1, :] --> 100, 2 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 2
        return out

face_dataset = utils.load_object("")






