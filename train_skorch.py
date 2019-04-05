# -*- coding: utf-8 -*-

import utils
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from model.net_arousal import myLSTM_arousal
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier

# configure the model dimension
input_dim = 1
hidden_dim = 100
layer_dim = 1
output_dim = 3
# Number of steps to unroll
seq_dim = 100 

# prepare input data
face_dataset = utils.load_object("data_1_5.pkl")
# create np array 
data_PD = []
data_FAP = []
target_arousal = []
target_valence = []
for i in range(len(face_dataset)):
    data_PD.append(face_dataset[i]['PD_avg_filtered'])
    data_FAP.append(face_dataset[i]['faceFAP'])
    target_arousal.append(face_dataset[i]['arousal'])
    target_valence.append(face_dataset[i]['valence'])
data_PD = np.array(data_PD, dtype=np.float32)
data_PD = np.reshape(data_PD,(-1,seq_dim,input_dim))
data_FAP = np.array(data_FAP, dtype=np.float32)
target_arousal = np.array(target_arousal, dtype=np.int64)
target_valence = np.array(target_valence, dtype=np.int64)


# split train test data
X_train , X_test, y_train, y_test = train_test_split(data_PD,target_arousal,test_size=0.2,random_state=42)


# visualize class
#pd.DataFrame(target_arousal).hist()
#pd.DataFrame(target_arousal).hist()
#pd.DataFrame(target_valence).hist()

# Instantiate neural net
# Definition : NeuralNetClassifier(module, criterion=torch.nn.NLLLoss, 
#                                train_split=CVSplit(5, stratified=True), *args, **kwargs)
net = NeuralNetClassifier(
        module=myLSTM_arousal,
        module__input_dim=input_dim,
        module__hidden_dim=hidden_dim,
        module__layer_dim=layer_dim,
        module__output_dim=output_dim,
        criterion=nn.CrossEntropyLoss,
        lr=0.05,
        max_epochs=10,
        device='cuda')

# fit model
net.fit(X_train,y_train)







