#%% get data for valence
import utils
import preprocessing.pre_utils as pu
import numpy as np

data_df = utils.load_object('faps_for_train.pkl')
valence = utils.load_object('valence.pkl')
valence_list = valence['valence'].tolist()

data_df = pu.match_label_with_sample(data_df,valence_list)
data_df = data_df.drop_duplicates('ori_idx')
           
label = data_df['label'].values.astype(np.int64)
data = data_df.drop(columns=['ori_idx','label']).values.astype(np.float32)
ori_idx = data_df['ori_idx'].tolist()

#X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0,random_state=42)

#%% get model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
filename = 'save_model/valence_svm.pkl'
loaded_model = joblib.load(filename)
valence_predicted = loaded_model.predict(data).tolist()
print(confusion_matrix(label, valence_predicted))
print('accuracy: {:.2f}'.format(accuracy_score(label, valence_predicted)))
val_np = np.array([valence_predicted,ori_idx]).transpose()
pred_val_df = pd.DataFrame(val_np,columns=['valence','ori_idx'])

#%% metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# ROC curve
fpr, tpr, thresholds = roc_curve(label, valence_predicted, pos_label=2) 

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
plt.title('Receiver operating characteristic (ROC) of Valence Classifier')
plt.legend(loc="lower right")
plt.show()


#%% merge arousal and valence data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

arousal_data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')
arousals_list = arousals['arousal'].tolist()

arousal_data_df = pu.match_label_with_sample(arousal_data_df,arousals_list)
arousal_data_df = arousal_data_df.set_index('ori_idx')
pred_val_df = pred_val_df.set_index('ori_idx')

arousal_data_df['valence'] = pred_val_df['valence']
arousal_data_df = arousal_data_df.dropna()

valence_predicted = loaded_model.predict(data).tolist()
print(confusion_matrix(label, valence_predicted))
print('accuracy: {:.2f}'.format(accuracy_score(label, valence_predicted)))

           
label = arousal_data_df['label'].values.astype(np.int64)
data = arousal_data_df[['slope_qr','delta_pq','delta_qr','valence']].values.astype(np.float32)

# split train test data
#X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.1)
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=42)

classifier = SVC(kernel='rbf',C=1,random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#%% metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# confusion metrix
print(confusion_matrix(y_test, y_pred))
print('accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
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
plt.title('Receiver operating characteristic (ROC) of Arousal Classifier')
plt.legend(loc="lower right")
plt.show()

#%% model persistence
from sklearn.externals import joblib
filename = 'save_model/combine_svm.pkl'
joblib.dump(classifier, filename , compress = 1)
# load the model from disk
loaded_model = joblib.load(filename)
acc = loaded_model.score(X_test, y_test)
print(acc)








