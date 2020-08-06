# RecommenderSystem_lightFM :blush:
## Overview:
Recommender systems are a very popular topic in e-commerce. They are frameworks and engines which help us to implement a recommender system easier. One of these engines is [LightFM](https://making.lyst.com/lightfm/docs/home.html) which is a satisfiable python framework.
## Goal:
The goal is to implement a recommender system with LightFM with 3 of [Kaggle](https://www.kaggle.com/) datasets.
3 data set and their related files are:
### 1.[Restaurant](https://www.kaggle.com/uciml/restaurant-data-with-consumer-ratings/metadata)   
files:
- collabrative_restaurant.py (main file)
- geoplaces2.json
- rating_final_U1011.json
- userprofile.json
- resturant_json.py
### 2.Book
files:
- book.py (main file)
- test_train.py
- book_wrap_vs_BPR.py 
- book_eval_K_OS.py
- mainbookup_lim1.json (10000 rows of main data in randomly picked)
### 3.[Size for clothes](https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation/metadata)  
files:
- size_test&trainWRAPvsBPR.py
- size_test&trainWRAP.py
- random_pick.py
- renttherunway_lim.json (10000 rows of main data in randomly picked)
- sizeRs.py (main file)
## Installation:
I used **Conda** in **Pycharm** and install **LightFM** with:
```
conda install -c conda-forge lightfm
conda install -c conda-forge/label/gcc7 lightfm
conda install -c conda-forge/label/cf201901 lightfm
conda install -c conda-forge/label/cf202003 lightfm 
```
## Data and LightFM:
One of the most important challenges is how to give the data to the package. First, read the Json file and create a dataset of lightfm :
```python
f = open('rating_final_U1011.json', )
ff = open('userprofile.json', )
df = open(r'geoplaces2.json')
data_User = json.load(ff)
data_item = json.load(df)
data = json.load(f)
dataset = Dataset()
```
Then fit the dataset with your data:
```python
dataset.fit((x['userID'] for x in data),
            (x['placeID'] for x in data), (x['budget'] for x in data_User),(x['price'] for x in data_item))
```
Now it's possible to create the matrixes:
```python
(interactions, weights) = dataset.build_interactions(((x['userID'], x['placeID']) for x in data))
print(repr(interactions))
user_interactions = dataset.build_user_features((x['userID'], [x['budget']]) for x in data_User)
print(repr(user_interactions))
item_interactions = dataset.build_item_features((x['placeID'], [x['price']]) for x in data_item)
print(repr(item_interactions))
```
This package is a model base package define and fit package:
```python
alpha = 1e-05
epochs = 70
num_components = 32
model = LightFM(no_components=num_components,
                loss='warp',
                learning_schedule='adadelta',
                user_alpha=alpha,
                item_alpha=alpha)
```
For testing and validating the model you need to split data to test and train like in test_train.py.  
Testing **learning_schedule** adadelta vs adagrad for cloth dataset:  
<img width="448" alt="Screenshot (994)" src="https://user-images.githubusercontent.com/41547574/89573678-36f8c100-d840-11ea-892b-8c2a2d2f9ef1.png">

Testing **loss** WARP vs BPR for cloth dataset:

<img width="448" alt="Screenshot (995)" src="https://user-images.githubusercontent.com/41547574/89574064-cbfbba00-d840-11ea-95ab-783682d9ef5a.png">

## Resourses:
http://www2.informatik.uni-freiburg.de/~cziegler/BX/  
https://making.lyst.com/lightfm/docs/home.html  
https://github.com/lyst/lightfm  

## Support:
Reach out to me at ghazalze@yahoo.com.  
Thanks [@alirezaomidi](https://github.com/alirezaomidi) :sweat_smile:
