import json
import random

import numpy as np
import scipy.sparse as sp
from lightfm.data import Dataset

fr = open('renttherunway_final_data_new.json', )
data = json.load(fr)
x = random.choices(data,k=10000)

with open('renttherunway_limited.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
