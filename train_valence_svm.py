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
from sklearn.metrics import accuracy_score

#%% get data
data_df = utils.load_object('faps_for_train.pkl')
valence = utils.load_object('valence.pkl')
arousals = utils.load_object('arousal.pkl')
valence_list = valence['valence'].tolist()
arousal_list = arousals['arousal'].tolist()

data_df = pu.match_label_with_sample(data_df,valence_list)
data_df = pu.match_label_with_sample(data_df,arousal_list,col_name='arousal')
           
label = data_df['label'].values.astype(np.int64)
data = data_df.drop(columns=['ori_idx','label','arousal']).values.astype(np.float32)

#test_df = utils.load_object('valence_for_train.pkl')
#test_df = pu.match_label_with_sample(test_df,valence_list)
#
#label_test = test_df['label'].values.astype(np.int64)
#data_test = test_df[['slope_qr','delta_pq','delta_qr']].values.astype(np.float32)

# split train test data
#X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.1)
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)

#%% train indepentdently
#classifier = SVC(kernel='rbf',C=1,random_state=0)
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
#%% Grid Search
classifier = SVC(random_state = 0)
# randomize hyperparameter search
parameters = {'kernel':('linear', 'rbf','poly'),
              'C':[1,3,5,7],
              'gamma':['auto']}
# micro average should be preferred over macro in case of imbalanced datasets
# now what metric to use to choose the best classifier from grid search
gs = GridSearchCV(classifier,parameters,cv=10,scoring=['accuracy','f1_micro'],
                  refit='accuracy',return_train_score=True)

# fit model using randomizer
gs.fit(X_train,y_train);

#%% report
utils.report(gs.cv_results_,3)

#%%
# Predicting the Test set results
y_pred = gs.best_estimator_.predict(X_test)

# confusion metrix
print(confusion_matrix(y_test, y_pred))
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2) 

roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (auc = %0.2f) , threshold = 0' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example  (ROC) of Separate Valence Classifier')
plt.legend(loc="lower right")
plt.show()

#%%
## Predicting the Test set results
#y_pred_test = gs.best_estimator_.predict(data_test)
#
## confusion metrix
#print('test with all samples')
#print(confusion_matrix(label_test, y_pred_test))
#print('accuracy: {:.2f}'.format(accuracy_score(label_test, y_pred_test)))
## ROC curve
#fpr, tpr, thresholds = roc_curve(label_test, y_pred_test, pos_label=2) 
#
#roc_auc = auc(fpr, tpr)
#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example (with all samples)')
#plt.legend(loc="lower right")
#plt.show()
#%% model persistence
from sklearn.externals import joblib
filename = 'save_model/valence_svm.pkl'
joblib.dump(gs.best_estimator_, filename , compress = 1)
# load the model from disk
loaded_model = joblib.load(filename)
acc = loaded_model.score(X_test, y_test)
print(acc)
