#%%
#import utils_dummy
import utils
import numpy as np
import preprocessing.pre_utils as pu
#import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt



#%% get data
data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')

match_arousal_list = pu.match_with_sample(arousals,data_df['ori_idx'])
data_df = data_df.reset_index(drop=True)
data_df = data_df.drop(columns=['ori_idx'])
data_df['arousal'] = match_arousal_list
            
label = data_df['arousal'].values.astype(np.int64)
#data = data.drop(columns=['arousal']).values.astype(np.float32)
data = data_df[['mean','median','max','min','skew']].values.astype(np.float32)

# split train test data
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)

#%% Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%% train indepentdently
classifier = SVC(kernel='rbf',C=7,random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
#%% Grid Search
classifier = SVC(random_state = 0)
# randomize hyperparameter search
parameters = {'kernel':('linear', 'rbf','poly','sigmoid'),
              'C':[1,3,5,7,10]}
# micro average should be preferred over macro in case of imbalanced datasets
# now what metric to use to choose the best classifier from grid search
gs = GridSearchCV(classifier,parameters,cv=2,scoring=['accuracy','f1_micro'],
                  refit='f1_micro',return_train_score=True)

# fit model using randomizer
gs.fit(X_train,y_train);

#%% report
utils.report(gs.cv_results_,5)

#%%
# Predicting the Test set results
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
