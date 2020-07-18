import os
import zipfile
import csv
import pandas as pd
import requests

import json
from itertools import islice
import sklearn.preprocessing
from lightfm.data import Dataset
import numpy as np
from lightfm import LightFM, lightfm
from lightfm.evaluation import auc_score
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# *********************************************************************
def create_item_dict(df, id_col, name_col,author_col):
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
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col] +' : '+df.loc[i, author_col]
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

def sample_recommendation_user(model, interactions, user_id, user_dict,
                               item_dict, threshold=0, nrec_items=10, show=True):
    '''
    Function to produce user recommendations
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output -
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    known_items = list(pd.Series(interactions.loc[user_id, :] \
                                     [interactions.loc[user_id, :] == threshold].index) \
                       .sort_values(ascending=False))

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter += 1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter += 1
    return return_score_list

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input -
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
# ***********************************************************************************************
fr = open('mainbookup_lim1.json', )
data = json.load(fr)
dataset = Dataset()
dataset.fit_partial((x['User-ID'] for x in data), (x['ISBN'] for x in data), None,
                    (x['Author'] for x in data))
(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN']) for x in data))
print(repr(interactions))
item_features = dataset.build_item_features(((x["ISBN"], [x["Author"]]) for x in data), normalize=True)
print(repr(item_features))
alpha = 1e-05
epochs = 70
num_components = 32
model = LightFM(no_components=num_components,
                loss='bpr',
                learning_schedule='adadelta',
                user_alpha=alpha,
                item_alpha=alpha)
model.fit(interactions, item_features=item_features)
df = pd.read_json(r'mainbookup_lim1.json')
interactions1 = create_interaction_matrix(df, "User-ID", "ISBN", "Author", norm=False, threshold=None)

item_dict = create_item_dict(df, "ISBN", "Title", "Author")
user_dict = create_user_dict(interactions1)
sample_recommendation_user(model, interactions1, 178622, user_dict,
                           item_dict, threshold="Rickey Singleton", nrec_items=10, show=True)
#model.item_embeddings
#item_em_matrix = create_item_emdedding_distance_matrix(model, interactions1)
#item_item_recommendation(item_em_matrix, "312970633", item_dict, n_items=10, show=True)
