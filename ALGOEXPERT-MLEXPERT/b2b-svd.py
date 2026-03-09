'''
Goal: incentivize SMBs to apply for biz CCs
Visa - has transDB - transFreq

Goal: discover the top10 ENTs Visa should partner with to entice SMBs
https://www.algoexpert.io/machine-learning/coding-questions/b2b-svd

'''

from sklearn.decomposition import TruncatedSVD
import numpy as np

'''
(sparse) avg_trxn_amount = #-trans conducted per ENT
(sparse) num_monthly_trxn = avg amount per trans per ENT

Goal = avg total monthly amount
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

Latent pattern learning
15 minutes and done ok
'''

def b2b_svd(avg_trxn_amount, num_monthly_trxn):
    avg_total_monthly_trans_amount = avg_trxn_amount * num_monthly_trxn # el-wise mult
    df_transposed = avg_total_monthly_trans_amount.T
    TARGET_NUM_ENT = 10
    svd = TruncatedSVD(n_components=TARGET_NUM_ENT)
    reduced_answer = svd.fit_transform(df_transposed) # fit to ENTERPRISE
    top10Indices = reduced_answer.argmax(axis=0) # indices with highest embedding values
    top10Columns = avg_trxn_amount.iloc[:, top10Indices]
    return top10Columns
