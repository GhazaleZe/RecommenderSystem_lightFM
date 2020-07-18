import json

import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM, lightfm
import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score, precision_at_k


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


def create_item_dict(df, id_col, name_col, size_col):
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
        item_dict[(df.loc[i, id_col])] = str(df.loc[i, name_col]) + " : " + str(df.loc[i, str(size_col)])
    return item_dict


def sample_recommendation_item(model, interactions, item_id, user_dict, item_dict, number_of_user):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id), n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list


def _shuffle(uids, iids, data, random_state):
    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)

    return (uids[shuffle_indices],
            iids[shuffle_indices],
            data[shuffle_indices])


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    This function takes an interaction set and splits it into
    two disjoint sets, a training set and a test set. Note that
    no effort is made to make sure that all items and users with
    interactions in the test set also have interactions in the
    training set; this may lead to a partial cold-start problem
    in the test set.

    Parameters
    ----------

    interactions: a scipy sparse matrix containing interactions
        The interactions to split.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (scipy.sparse.COOMatrix,
                    scipy.sparse.COOMatrix)
         A tuple of (train data, test data)
    """

    if not sp.issparse(interactions):
        raise ValueError('Interactions must be a scipy.sparse matrix.')

    if random_state is None:
        random_state = np.random.RandomState()

    interactions = interactions.tocoo()

    shape = interactions.shape
    uids, iids, data = (interactions.row,
                        interactions.col,
                        interactions.data)

    uids, iids, data = _shuffle(uids, iids, data, random_state)

    cutoff = int((1.0 - test_percentage) * len(uids))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = sp.coo_matrix((data[train_idx],
                           (uids[train_idx],
                            iids[train_idx])),
                          shape=shape,
                          dtype=interactions.dtype)
    test = sp.coo_matrix((data[test_idx],
                          (uids[test_idx],
                           iids[test_idx])),
                         shape=shape,
                         dtype=interactions.dtype)

    return train, test


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
            print(str(counter) + '- ' + str(i))
            counter += 1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + str(i))
            counter += 1
    return return_score_list


fr = open('renttherunway_lim.json', )
data = json.load(fr)
dataset = Dataset()
dataset.fit((x['user_id'] for x in data), (x['item_id'] for x in data), (x['fit'] for x in data),
            (x['size'] for x in data))
# num_users, num_items = dataset.interactions_shape()
# print('Num users: {}, num_items {}.'.format(num_users, num_items))
(interactions, weights) = dataset.build_interactions(((x['user_id'], x['item_id']) for x in data))
print(repr(interactions))
user_feature = dataset.build_user_features(((x['user_id'], [x['fit']]) for x in data))
print(repr(user_feature))
item_feature = dataset.build_item_features(((x['item_id'], [x['size']]) for x in data))
print(repr(item_feature))
(train, test) = random_train_test_split(interactions, test_percentage=0.2, random_state=None)
alpha = 1e-3
epochs = 70

model = LightFM(no_components=30,
                loss='warp',
                learning_schedule='adagrad',
                user_alpha=alpha,
                item_alpha=alpha)

model.fit(train, user_feature, item_feature, epochs=20, num_threads=1)
# predictions = model.predict([93937], [1108814], num_threads=1)
'''train_precision = precision_at_k(model, train, k=10, user_features=user_feature, item_features=item_feature).mean()
test_precision = precision_at_k(model, test, k=10, user_features=user_feature, item_features=item_feature).mean()

train_auc = auc_score(model, train, user_features=user_feature, item_features=item_feature).mean()
test_auc = auc_score(model, test, user_features=user_feature, item_features=item_feature).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))'''

df = pd.read_json(r'renttherunway_lim.json')
interactions1 = create_interaction_matrix(df, "user_id", "item_id", "rating", norm=False, threshold=None)
user_dict = create_user_dict(interactions1)
item_dict = create_item_dict(df, "item_id", "item_id", "size")
# x = sample_recommendation_item(model, interactions1, 1108814, user_dict, item_dict, 4)
# for i in x:
#    print(i)
sample_recommendation_user(model, interactions1, 907115, user_dict,
                           item_dict, threshold=7, nrec_items=5, show=True)

