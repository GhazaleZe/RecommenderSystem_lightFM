
import pandas as pd
import json

#create jason file of rating_final
df = pd.read_csv(r'rating_final - Only.csv')
# df.to_json(r'rating_final1.json',orient='index')
df.to_json(r'rating_final - Only.json', orient='records')

f = open('rating_final - Only.json', )

# returns JSON object as
# a dictionary
data = json.load(f)

for i in data:
    print(i)


df = pd.read_csv(r'geoplaces2.csv')
df.to_json(r'geoplaces2.json', orient='records')

# returns JSON object as
# a dictionary
restaurant = json.load(df)
# Iterating through the json
# list
for i in restaurant:
    print(i)


# Closing file
f.close()
df.close()
