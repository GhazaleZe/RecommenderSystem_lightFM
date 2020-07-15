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

# restaurant_metadata = pd.read_json('rating_final.json', lines=True)
from scipy import sparse

f = open('rating_final_U1011.json', )

# returns JSON object as
# a dictionary
data = json.load(f)

# returns JSON object as
# a dictionary
dataset = Dataset()
dataset.fit((x['userID'] for x in data),
            (x['placeID'] for x in data))

# print(dataset)
(interactions, weights) = dataset.build_interactions(((x['userID'], x['placeID']) for x in data))
# print(repr(interactions))
# print(interactions)
# print(test_interactions)
NUM_THREADS = 1
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

# Let's fit a WARP model: these generally have the best performance.
model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
                no_components=NUM_COMPONENTS)

# Run 3 epochs and time it.
model = model.fit(interactions, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

from lightfm.evaluation import auc_score

model.item_biases *= 0.0



# ************************************************************************************

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


# ****************************************************************************************
def create_item_dict(df, id_col, name_col):
    """
    Function to create an item dictionary based on their item_id and item name
    Required Input -
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    """
    item_dict = {}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
    return item_dict


# *******************************************************************************

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


# **************************************************************************************

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
                                     [interactions.loc[user_id, :] > threshold].index) \
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


df = pd.read_json(r'geoplaces2.json')
dfm = pd.read_json(r'rating_final_U1011.json')
interactions = create_interaction_matrix(dfm, "userID", "placeID", "rating", norm=False, threshold=None)
item_dict = create_item_dict(df, "placeID", "name")
user_dict = create_user_dict(interactions)
sample_recommendation_user(model, interactions, "U1011", user_dict,
                           item_dict, threshold=0, nrec_items=5, show=True)
#print(item_dict[135072])
f.close()
