# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:24:05 2022

@author: Sai pranay
"""
#----------------------importing the data set----------------------------------
import pandas as pd
import numpy as np
bks = pd.read_csv("E:\DATA_SCIENCE_ASS\RECOMMDATION SYSTEM\\book.csv", encoding="ISO-8859-1")
bks.shape
list(bks)
bks.head()
list(bks)


bks['User.ID']
bks['Book.Rating']
bks['Book.Title']

# droping

bks.drop(bks.columns[[0]],axis=1,inplace=True)

bks.sort_values('User.ID')

#number of unique users in the dataset
len(bks)
len(bks['User.ID'].unique())
len(bks['Book.Title'].unique())

bks['Book.Rating'].value_counts()
bks['Book.Rating'].hist()

list(bks)
bks.shape


user_bks=bks.pivot_table(index ='User.ID',columns ='Book.Title',values ='Book.Rating')
pd.crosstab(bks['User.ID'],bks['Book.Title'])
user_bks
user_bks.iloc[0]
user_bks.iloc[200]
list(user_bks)

#Impute those NaNs with 0 values
user_bks.fillna(0, inplace=True)

user_bks

# from scipy.spatial.distance import cosine correlation
#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances( user_bks.values,metric='cosine')
user_sim

#user_sim = 1 - pairwise_distances( user_movies_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df
#Set the index and column names to user ids 
user_sim_df.index   = bks['User.ID'].unique()
user_sim_df.columns = bks['User.ID'].unique()

user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:5]

bks[(bks['User.ID']==276729) | (bks['User.ID']==276726)]

user_276729=bks[bks['User.ID']==276729]
user_276729

user_276726=bks[bks['User.ID']==276726]
user_276726

user_276726=bks[bks['User.ID']==276726]
user_276726

user_276736=bks[bks['User.ID']==276736]
user_276736

pd.merge(user_276726,user_276736,on='Book.Rating',how='outer')

