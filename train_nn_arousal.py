#%%
#import utils_dummy
import utils
import torch
import torch.nn as nn
import numpy as np
import preprocessing.pre_utils as pu
#import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from skorch import NeuralNetClassifier
from model.dummy_dataset_class import DummyDataset
import matplotlib.pyplot as plt

#%%
# Standard plotly imports
import plotly
import plotly.io as pio
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#%%

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
####### --------- train ---------------###########

#%% get data
data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')

match_arousal_list = pu.match_with_sample(arousals,data_df['ori_idx'])
data_df = data_df.reset_index(drop=True)
data_df = data_df.drop(columns=['ori_idx'])
data_df['arousal'] = match_arousal_list
            
label = data_df['arousal'].values.astype(np.int64)
#data = data.drop(columns=['arousal']).values.astype(np.float32)
data = data_df[['delta_pq','delta_qr','slope_qr']].values.astype(np.float32)

# split train test data
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)


#%% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%% visualize data
fig = data_df['arousal'].iplot(
        kind='histogram',
        title='arousal class',
        yTitle='% of samples in each class',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)

#%%

# configure the model dimension
input_dim = data.shape[1]
hidden_dim = 20
output_dim = 2

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
        batch_size=10,
        device='cuda')

## fit model
#net.fit(X_train,y_train)

#%%
# randomize hyperparameter search
lr = [0.005,0.05,0.5]
params = {
    'optimizer__lr': lr,
    'max_epochs':[35,100,150],
    'module__hidden_dim':[45,50,55]
#    'module__num_units': [14,20,28,36,42],
#    'module__drop' : [0,.1,.2,.3,.4]
}
# micro average should be preferred over macro in case of imbalanced datasets
# now what metric to use to choose the best classifier from grid search
gs = GridSearchCV(net,params,cv=3,scoring=['f1_micro','accuracy'],
                  refit='accuracy',return_train_score=True)

# fit model using randomizer
gs.fit(X_train,y_train);
#%%
# review top 10 results and parameters associated
utils.report(gs.cv_results_,3)

#%%
## predict
#y_pred_prob = net.predict_proba(X_test)
#print(y_pred_prob)
#y_pred = net.predict(X_test)
#print(y_pred)
#%%
#########------------ Evaluate model ----------#############
# predict on test data
y_pred = gs.best_estimator_.predict(X_test)

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


