import os
import zipfile
import csv
import pandas as pd
import requests

import json
from itertools import islice

import sklearn.preprocessing
from lightfm.data import Dataset

import pandas
import numpy as np
from lightfm import LightFM
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

''''
#create json file from csv file
df = pd.read_csv(r'books_with_blurbs.csv')
df.to_json(r'books_with_blurbs.json', orient='records')
'''


# *********************************************************************
def create_item_dict(df, id_col, name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input -
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict = {}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
    return item_dict


# *****************************************************************************************
def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output -
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
        .sum().unstack().reset_index(). \
        fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions


# ************************************************************************************
def create_item_emdedding_distance_matrix(model, interactions):
    '''
    Function to create item-item distance embedding matrix
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
    Expected Output -
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
    '''
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix


# *****************************************************************************
def item_item_recommendation(item_emdedding_distance_matrix, item_id,
                             item_dict, n_items=10, show=True):
    '''
    Function to create item-item recommendation
    Required Input -
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
        - item_id  = item ID for which we need to generate recommended items
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - n_items = Number of items needed as an output
    Expected Output -
        - recommended_items = List of recommended items
    '''
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id, :]. \
                                       sort_values(ascending=False).head(n_items + 1). \
                                       index[1:n_items + 1]))
    if show == True:
        print("Item of interest :{0}".format(item_dict[item_id]))
        print("Item similar to the above item:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' + item_dict[i])
            counter += 1
    return recommended_items


# *********************************************************************************
def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30, n_jobs=4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run
        - n_jobs = number of cores used for execution
    Expected Output  --
        Model - Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components=n_components, loss=loss, k=k)
    model.fit(x, epochs=epoch, num_threads=n_jobs)
    return model


# ***********************************************************************************************
fr = open('BxUB.json', )
data = json.load(fr)
dataset = Dataset()
dataset.fit((x['User-ID'] for x in data), (x['ISBN'] for x in data), (x['Book-Rating'] for x in data),
            (x['Author'] for x in data))
(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN']) for x in data))
print(repr(interactions))
item_features = dataset.build_item_features(((x["ISBN"], [x["Author"]]) for x in data))
print(repr(item_features))

'''

data = json.load(fr)
df = pd.read_json(r'BxUB.json')
item_dict = create_item_dict(df, "ISBN", "Title")
dataset = Dataset()

(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN']) for x in data))
print(repr(interactions))
#print("hello")

fr = open('Mybook_user.json', )
data_user = json.load(fr)
df = pd.read_json(r'Mybook_user.json')
item_dict = create_item_dict(df, "ISBN", "Title")
dataset = Dataset()
dataset.fit((x['User-ID'] for x in data_user), (x['ISBN'] for x in data_user), None, (x['Author'] for x in data_user))
(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN']) for x in data_user))
print(repr(interactions))
ISBN, Blurb = dataset.interactions_shape()
y = 'Title: {}, Blurb {}.'.format(ISBN, Blurb)
print(y)
print("hello")
item_features = dataset.build_item_features(((x["ISBN"], [x["Author"]]) for x in data_user))
# interactions = dataset.build_interactions(((x["Title"], x["Blurb"]) for x in data))
model = LightFM(loss='bpr')
model.fit(interactions, item_features=item_features)
item_emdedding_distance_matrix = create_item_emdedding_distance_matrix(model, interactions)

#item_item_recommendation(item_emdedding_distance_matrix, "440220653",
#                             item_dict, n_items=10, show=True)

print("hello")
'''
''''
model = LightFM(loss='bpr')
model.fit(interactions, item_features=item_features)
print(repr(item_features))

df=pd.read_json(r'books_with_blurbs.json')
interactions = create_interaction_matrix(df, "Title", "Blurb", norm=False, threshold=None)
model = runMF(interactions, n_components=30, loss='bpr', k=15, epoch=30, n_jobs=4)
item_emdedding_distance_matrix = create_item_emdedding_distance_matrix(model, interactions)
item_dict=create_item_dict(df, "ISBN", "Title")
item_item_recommendation(item_emdedding_distance_matrix, "Goodbye to the Buttermilk Sky",
                             item_dict, n_items=10, show=True)
'''
