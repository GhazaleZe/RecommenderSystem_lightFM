
import json
import time
from lightfm.data import Dataset
from lightfm import LightFM, lightfm
import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score


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
fr = open('parttialbook.json', )
data = json.load(fr)
dataset = Dataset()
dataset.fit((x['User-ID'] for x in data), (x['ISBN'] for x in data), (x['Book-Rating'] for x in data),
            (x['Author'] for x in data))
(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN']) for x in data))
print(repr(interactions))

(train, test) = random_train_test_split(interactions, test_percentage=0.2, random_state=None)

alpha = 1e-05
epochs = 70
num_components = 32

warp_model = LightFM(no_components=num_components,
                    loss='warp',
                    learning_schedule='adadelta',
                    max_sampled=100,
                    user_alpha=alpha,
                    item_alpha=alpha)

bpr_model = LightFM(no_components=num_components,
                    loss='bpr',
                    learning_schedule='adadelta',
                    user_alpha=alpha,
                    item_alpha=alpha)

warp_duration = []
bpr_duration = []
warp_auc = []
bpr_auc = []

for epoch in range(epochs):
    start = time.time()
    warp_model.fit_partial(train, epochs=1)
    warp_duration.append(time.time() - start)
    warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

for epoch in range(epochs):
    start = time.time()
    bpr_model.fit_partial(train, epochs=1)
    bpr_duration.append(time.time() - start)
    bpr_auc.append(auc_score(bpr_model, test, train_interactions=train).mean())
x = np.arange(epochs)
plt.plot(x, np.array(warp_auc))
plt.plot(x, np.array(bpr_auc))
plt.legend(['WARP AUC', 'BPR AUC'], loc='upper right')
plt.show()
