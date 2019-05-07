# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:47:57 2019

@author: Sheikh Rabiul Islam
Purpose: frequent pattern (variables) mining.
 
"""
import pandas as pd
import numpy as np

file_name = "output4-clean.csv"
dataset = pd.read_csv(file_name, sep=',')
dataset_bk = pd.read_csv(file_name, sep=',')

dataset.drop(['id','support','length'], axis=1, inplace = True)

#heuristics_feature = set(['debtToIncomeRatioOriginal','DebtToIncomeRatioCurrent','creditScore','creditScoreCoborrower','creditScoreOriginal','LTV','LTVoriginal','CLTV','CLTVoriginal','UPBactual', 'UPBoriginal','propertyState', 'postalCode','interestRateCurrent','interestRateOriginal'])

hf1 = set(['DebtToIncomeRatioOriginal','DebtToIncomeRatioCurrent','currentLoanDelinquencyStatus']) #capacity , we might include payment to income too
hf2 = set(['creditScore','creditScoreCoborrower','creditScoreOriginal']) #character
hf3 = set(['LTV','LTVoriginal','CLTV','CLTVoriginal']) #collateral
hf4 = set(['UPBactual', 'UPBoriginal']) #capital
hf5 = set(['propertyState', 'postalCode','interestRateCurrent','interestRateOriginal']) #conditions


i_size = len(dataset)
j_size = dataset.shape[1]
index_l = []

for i in range(i_size):
    #print("i",i)
    s = set()
    for j in range(j_size):
        #print("j",j)
        #flag2 = 0
        if dataset.iloc[i][j]  in hf1:
            s.add(1)
        if dataset.iloc[i][j]  in hf2:
            s.add(2)
        if dataset.iloc[i][j]  in hf3:
            s.add(3)
        if dataset.iloc[i][j]  in hf4:
            s.add(4)
        if dataset.iloc[i][j]  in hf5:
            s.add(5)

    flag = 0
    #print(s)
    for k in range(1,6):
        #print("k",k)
        if k not in s:
            flag = flag+1
            #print("flag ", flag)
            
    if flag > 0:
        index_l.append(i)


dataset = dataset.drop(dataset.index[index_l])
dataset.to_csv("output4-clean-filtered.csv", sep=',')

print(dataset)