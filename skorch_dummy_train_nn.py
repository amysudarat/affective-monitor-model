# -*- coding: utf-8 -*-
#%%
#import utils_dummy
import utils
import torch
import torch.nn as nn
import numpy as np
#import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from skorch import NeuralNetClassifier
from model.dummy_dataset_class import DummyDataset
import matplotlib.pyplot as plt

#import torch.nn.functional as F


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
        self.fc3 = nn.Linear(hidden_dim, output_dim)
#        # Non-linearity 3
#        self.sigmoid3 = nn.Sigmoid()
#        
#        # Linear function 4 (readout): 100 --> 10
#        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
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
#        # Non-linearity 2
#        out = self.sigmoid3(out)
#        
#        # Linear function 4 (readout)
#        out = self.fc4(out)
        return out


#%%
####### ---------- utility function ------ #########
# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_f1_micro'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_accuracy'][candidate],
                  results['std_test_accuracy'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#%%

####### --------- train ---------------###########

dummy_dataset = DummyDataset()

# prepare data
data = []
label = []

for i in range(len(dummy_dataset)):
    data.append(dummy_dataset[i]['data'])    
    label.append(dummy_dataset[i]['label'])
    
#data = utils.load_object('dummy_data.pkl')
#label = utils.load_object('dummy_label.pkl')
    
data = np.array(data, dtype=np.float32)
label = np.array(label, dtype=np.int64)



# split train test data
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)
## visualise
#plt.hist(y_train)
#plt.hist(y_test)

## normalize data to 0 mean and unit std
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# min max scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
# configure the model dimension
input_dim = 100
hidden_dim = 30
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
        optimizer__lr=0.005,
#        warm_start=True,
        max_epochs=30,
        device='cuda')

## fit model
#net.fit(X_train,y_train)

#%%
# randomize hyperparameter search
lr = [0.01,0.05,0.1]
params = {
    'optimizer__lr': lr,
    'max_epochs':[70,150,200],
#    'module__num_units': [14,20,28,36,42],
#    'module__drop' : [0,.1,.2,.3,.4]
}
# micro average should be preferred over macro in case of imbalanced datasets
# now what metric to use to choose the best classifier from grid search
gs = GridSearchCV(net,params,cv=2,scoring=['accuracy','f1_micro'],
                  refit='f1_micro',return_train_score=True)

# fit model using randomizer
gs.fit(X_train_scaled,y_train);
#%%
# review top 10 results and parameters associated
report(gs.cv_results_,5)

# get training and validation loss
epochs = [i for i in range(len(gs.best_estimator_.history))]
train_loss = gs.best_estimator_.history[:,'train_loss']
valid_loss = gs.best_estimator_.history[:,'valid_loss']
# plot learning curve
plt.figure()
plt.plot(epochs,train_loss,'g-');
plt.plot(epochs,valid_loss,'r-');
plt.title('Training Loss Curves');
plt.xlabel('Epochs');
plt.ylabel('Mean Squared Error');
plt.legend(['Train','Validation']);
plt.show()
#%%
## predict
#y_pred_prob = net.predict_proba(X_test)
#print(y_pred_prob)
#y_pred = net.predict(X_test)
#print(y_pred)
#%%
#########------------ Evaluate model ----------#############
# predict on test data
y_pred = gs.best_estimator_.predict(X_test_scaled)

# confusion metrix
print(confusion_matrix(y_test, y_pred))
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2) 
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#%%

