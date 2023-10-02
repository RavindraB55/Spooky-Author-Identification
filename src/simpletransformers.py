import os
import numpy as np 
import pandas as pd
import tensorflow as tf 

dir = './spooky-author-identification'

for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv(f"{dir}/train.csv")
print(data.head())

data['author_num'] = data["author"].map({'EAP':0, 'HPL':1, 'MWS':2})
print(data.head())

train = data.sample(frac=0.8,random_state=200) #random state is a seed value
test = data.drop(train.index)
