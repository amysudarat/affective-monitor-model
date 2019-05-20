#%% get data for valence
import utils
import preprocessing.pre_utils as pu
import numpy as np

data_df = utils.load_object('faps_for_train.pkl')
valence = utils.load_object('valence.pkl')
valence_list = valence['valence'].tolist()

data_df = pu.match_label_with_sample(data_df,valence_list)
           
label = data_df['label'].values.astype(np.int64)
data = data_df.drop(columns=['ori_idx','label']).values.astype(np.float32)
ori_idx = data_df['ori_idx'].tolist()

#X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0,random_state=42)

#%% get model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
filename = 'save_model/valence_svm.pkl'
loaded_model = joblib.load(filename)
valence_predicted = loaded_model.predict(data)
print(confusion_matrix(label, valence_predicted))
print('accuracy: {:.2f}'.format(accuracy_score(label, valence_predicted)))

#%% train arousal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

arousal_data_df = utils.load_object('pd_for_train.pkl')
arousals = utils.load_object('arousal.pkl')
arousals_list = arousals['arousal'].tolist()

arousal_data_df = pu.match_label_with_sample(arousal_data_df,arousals_list)
           
label = data_df['label'].values.astype(np.int64)
data = data_df[['slope_qr','delta_pq','delta_qr']].values.astype(np.float32)

test_df = utils.load_object('pd_for_test.pkl')
test_df = pu.match_label_with_sample(test_df,arousals_list)

label_test = test_df['label'].values.astype(np.int64)
data_test = test_df[['slope_qr','delta_pq','delta_qr']].values.astype(np.float32)

# split train test data
#X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.1)
X_train , X_test, y_train, y_test = train_test_split(data,label,test_size=0.8,random_state=42)

classifier = SVC(kernel='rbf',C=1,random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))







