# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:04:57 2019

@author: Sheikh Rabiul Islam
Purpose: frequent pattern (variables) mining.

"""
import pandas as pd
import numpy as np

file_name = "literature_review_template_tables.csv"
dataset = pd.read_csv(file_name, sep=',')

features = []
features_d = {}

#build the feature list
def build_features(dataset):
    for row_index in range(len(dataset)):
        row = dataset.iloc[row_index][0]
        row_list = row.split(',')
        
        for feature in row_list:
            feature = feature.strip()
            if feature not in features_d:
                features_d[feature]=0
                features.append(feature)

#call the function
build_features(dataset)
        

#making feature vector
dataset_e_df = pd.DataFrame(np.zeros((len(dataset),len(features_d)), dtype = 'int8'),columns =  features_d)



for row_index in range(len(dataset)):
    row = dataset.iloc[row_index][0]
    row_list = row.split(',')
    
    for feature in row_list:
        feature = feature.strip()
        #print(feature)
        dataset_e_df.at[row_index,feature] = 1


# print features and associated count        
print("feature","count") 
print(dataset_e_df.sum(axis = 0))

dataset_ee_df_alt = pd.DataFrame(dataset_e_df.sum(axis = 0),columns = ['frequency'])
dataset_ee_df_alt.to_csv("output1.csv",sep=",") 
dataset_ee_df_alt.to_csv("output1.csv",sep=",") 
#save the dataframe in to a csv file
dataset_e_df.to_csv("output2.csv",sep=",")


source: #apriory http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
from mlxtend.frequent_patterns import apriori
min_support =.05
max_len = 8
frequent_itemset = apriori(dataset_e_df, min_support=min_support,use_colnames=True, max_len=max_len, n_jobs=4)

frequent_itemset.to_csv("output3.csv",sep=",")

print(frequent_itemset)

length = 8
print("print frequent item set with length:", length)
frequent_itemset['length'] = frequent_itemset['itemsets'].apply(lambda x: len(x))
frequent_itemset = frequent_itemset[ (frequent_itemset['length'] == length) & (frequent_itemset['support'] >= min_support) ]
frequent_itemset.to_csv("output4.csv",sep=",")
print(frequent_itemset)