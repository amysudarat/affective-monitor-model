# -*- coding: utf-8 -*-

import utils
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier


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


####### --------- train ---------------###########
data_df = utils.load_object('pd_for_train.pkl')
data = utils.load_object('pd_for_train.pkl')
label = data['arousal'].values.astype(np.int64)
#data = data.drop(columns=['arousal']).values.astype(np.float32)
data = data[['mean','median','max','min','skew']].values.astype(np.float32)

# split train test data
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)

# calculate percentage of classes
portion, _ = np.histogram(label)
portion = (portion/portion.sum())*100


# visualize label
plt.hist(y_train)
plt.hist(y_test)

# configure the model dimension
input_dim = data.shape[1]
hidden_dim = 5
output_dim = 5

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
        lr=0.01,
        max_epochs=100,
        device='cuda')

# fit model
fit_param = net.fit(X_train,y_train)

# predict
y_pred_prob = net.predict_proba(X_test)
print(y_pred_prob)
y_pred = net.predict(X_test)
print(y_pred)


