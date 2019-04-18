# -*- coding: utf-8 -*-

import utils_dummy
import utils
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from model.dummy_dataset_class import DummyDataset
import matplotlib.pyplot as plt


###### --------- define net ------------###########
class simple_fnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(simple_fnn, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.sigmoid1 = nn.Sigmoid()
        
        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.sigmoid2 = nn.Sigmoid()
        
        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.sigmoid3 = nn.Sigmoid()
        
        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.sigmoid1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.sigmoid2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.sigmoid3(out)
        
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out


####### --------- train ---------------###########3

dummy_dataset = DummyDataset()

# prepare data
data = []
label = []

for i in range(len(dummy_dataset)):
    data.append(dummy_dataset[i]['data'])    
    label.append(dummy_dataset[i]['label'])
    
data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=np.int64)


#data = utils.load_object('dummy_data.pkl')
#label = utils.load_object('dummy_label.pkl')


# split train test data
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)
## visualise
plt.hist(y_train)
plt.hist(y_test)

# configure the model dimension
input_dim = 100
hidden_dim = 100
output_dim = 4

# Instantiate neural net
# Definition : NeuralNetClassifier(module, criterion=torch.nn.NLLLoss, 
#                                train_split=CVSplit(5, stratified=True), *args, **kwargs)
net = NeuralNetClassifier(
        module=simple_fnn,
        module__input_dim=input_dim,
        module__hidden_dim=hidden_dim,
        module__output_dim=output_dim,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.05,
        max_epochs=100,
        device='cuda')

# fit model
net.fit(X_train,y_train)

# predict
y_pred_prob = net.predict_proba(X_test)
print(y_pred_prob)
y_pred = net.predict(X_test)
print(y_pred)
