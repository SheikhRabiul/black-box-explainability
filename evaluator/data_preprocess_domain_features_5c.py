# Author: Sheikh Rabiul Islam
# Date: 07/13/2019; updated:  07/15/2019
# Purpose: load preprocessed data, delete records from a particular attack from the training set only. 
#		save the fully processed data as numpy array (binary: data/____.npy)

import pandas as pd   
import numpy as np
import time
start = time.time()

# import data
dataset_train = pd.read_csv('data/data_preprocessed_numerical_train.csv', sep=',')
dataset_test = pd.read_csv('data/data_preprocessed_numerical_test.csv', sep=',')

#drop extra columns
#feature_selected = ['ACK Flag Count','Active Mean','Active Min','Average Packet Size','Bwd IAT Mean',
#                    'Bwd Packet Length Std','Bwd Packets/s','Fwd IAT Mean','Fwd IAT Min','Fwd Packet Length Mean','Fwd Packets/s',
#                    'Fwd PSH Flags','Flow Duration','Flow IAT Mean','Flow IAT Min','Flow IAT Std','Init_Win_bytes_forward',
#                    'PSH Flag Count','Subflow Fwd Bytes','SYN Flag Count','Total Length of Fwd Packets', 'Class']

#pattern 'numberOfBorrowers', 'originalUPB', 'originalInterestRate', 'currentInterestRate', 'currentLoanDelinquencyStatus', 'originalCombinedLoanToValue','currentActualUPB', 'creditScore', 'defaulted'
feature_selected = ['originalUPB', 'originalInterestRate', 'currentInterestRate', 'currentLoanDelinquencyStatus', 'originalCombinedLoanToValue','currentActualUPB', 'creditScore', 'defaulted']

dataset_train = dataset_train[feature_selected]
dataset_test = dataset_test[feature_selected]
from sklearn import preprocessing

dataset_train_class = np.array(dataset_train['defaulted'])
xx = dataset_train.iloc[:,0:-1].values
xx_columns = dataset_train.columns
xx_columns = xx_columns[0:-1]
min_max_scaler = preprocessing.MinMaxScaler()  #scaling is done here [0,1]
xx_scaled = min_max_scaler.fit_transform(xx)
dataset_train = pd.DataFrame(xx_scaled)
df_part1 = pd.DataFrame(xx_scaled, columns=xx_columns)
df_part2 = pd.DataFrame(dataset_train_class, columns=['defaulted'])   
dataset_train = pd.concat([df_part1,df_part2], axis = 1) 


dataset_train.insert(loc = 7, column = 'character', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 8, column = 'capacity', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 9, column = 'capital', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 10, column = 'conditions', value = np.zeros(len(dataset_train)))
dataset_train.insert(loc = 11, column = 'collateral', value = np.zeros(len(dataset_train)))




dataset_train['character'] = dataset_train['character'] + dataset_train['creditScore']*-0.123535817903073       #correlation with defaulted -0.123535817903073

dataset_train['capacity'] = dataset_train['capacity'] + dataset_train['currentLoanDelinquencyStatus']*0.688183175505541  #correlation with defaulted 0.688183175505541

dataset_train['capital'] = dataset_train['capital'] + dataset_train['originalUPB']*0.0837925355305464  #correlation with defaulted 0.0837925355305464
dataset_train['capital'] = dataset_train['capital'] + dataset_train['currentActualUPB']*-0.731821113582467  #correlation with defaulted -0.731821113582467

dataset_train['conditions'] = dataset_train['conditions'] + dataset_train['originalInterestRate']*0.201568736339354  #correlation with defaulted 0.201568736339354
dataset_train['conditions'] = dataset_train['conditions'] + dataset_train['currentInterestRate']*-0.621300123845228 #correlation with defaulted  -0.621300123845228

dataset_train['collateral'] = dataset_train['collateral'] + dataset_train['originalCombinedLoanToValue']*0.324007308140558  #correlation with defaulted 0.324007308140558



dataset_test_class = np.array(dataset_test['defaulted'])
xx = dataset_test.iloc[:,0:-1].values
xx_columns = dataset_test.columns
xx_columns = xx_columns[0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
xx_scaled = min_max_scaler.fit_transform(xx)
dataset_test = pd.DataFrame(xx_scaled)
df_part1 = pd.DataFrame(xx_scaled, columns=xx_columns)
df_part2 = pd.DataFrame(dataset_test_class, columns=['defaulted'])   
dataset_test = pd.concat([df_part1,df_part2], axis = 1) 


dataset_test.insert(loc = 7, column = 'character', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 8, column = 'capacity', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 9, column = 'capital', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 10, column = 'conditions', value = np.zeros(len(dataset_test)))
dataset_test.insert(loc = 11, column = 'collateral', value = np.zeros(len(dataset_test)))


dataset_test['character'] = dataset_test['character'] + dataset_test['creditScore']*-0.123535817903073       #correlation with defaulted -0.123535817903073

dataset_test['capacity'] = dataset_test['capacity'] + dataset_test['currentLoanDelinquencyStatus']*0.688183175505541  #correlation with defaulted 0.688183175505541

dataset_test['capital'] = dataset_test['capital'] + dataset_test['originalUPB']*0.0837925355305464  #correlation with defaulted 0.0837925355305464
dataset_test['capital'] = dataset_test['capital'] + dataset_test['currentActualUPB']*-0.731821113582467  #correlation with defaulted -0.731821113582467

dataset_test['conditions'] = dataset_test['conditions'] + dataset_test['originalInterestRate']*0.201568736339354  #correlation with defaulted 0.201568736339354
dataset_test['conditions'] = dataset_test['conditions'] + dataset_test['currentInterestRate']*-0.621300123845228 #correlation with defaulted  -0.621300123845228

dataset_test['collateral'] = dataset_test['collateral'] + dataset_test['originalCombinedLoanToValue']*0.324007308140558  #correlation with defaulted 0.324007308140558



feature_selected = ['character','capacity','capital','conditions','collateral','defaulted']
dataset_train = dataset_train[feature_selected]
dataset_test = dataset_test[feature_selected]

dataset_bk = pd.concat([dataset_train,dataset_test], axis = 0) 
dataset_bk.to_csv("data/data_preprocessed_numerical_5c.csv", sep=',')

#dataset_train = dataset_train.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)
#dataset_test = dataset_test.drop(['Unnamed: 0', 'index', 'index_old', 'Class_all'], axis=1)

X_train = dataset_train.iloc[:,0:-1].values
y_train = dataset_train.iloc[:,-1].values

X_test = dataset_test.iloc[:,0:-1].values
y_test = dataset_test.iloc[:,-1].values

end = time.time()
print("checkpoint 1:", end-start)

#dump onehot encoded training data
# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_domain_features.npy',X_train)
np.save('data/data_fully_processed_y_train_domain_features.npy',y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_test_domain_features.npy',X_test)
np.save('data/data_fully_processed_y_test_domain_features.npy',y_test)


end = time.time()
print("checkpoint 2:", end-start)

################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_resampled_domain_features.npy',X_train_res)
np.save('data/data_fully_processed_y_train_resampled_domain_features.npy',y_train_res)

df_dump_part1 = pd.DataFrame(X_train_res, columns=dataset_train.iloc[:,0:-1].columns.values)
df_dump_part2 = pd.DataFrame(y_train_res, columns=['defaulted'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)
df_dump.to_csv("data/data_preprocessed_numerical_train_res_5c.csv",encoding='utf-8')

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

