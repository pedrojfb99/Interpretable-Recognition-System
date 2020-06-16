#Using the chefboost framework
#https://github.com/serengil/chefboost
from chefboost import Chefboost as chef

import numpy as np


import pandas as pd

import pickle


images = pickle.load(open("needed_correct_images.p","rb"))



def return_amount_of_samples(n_samples):



    chars = []
    second_chars = []
    labels = []


    for n in range(n_samples):
        chars.append(np.random.rand(1,9)[0])
        second_chars.append(np.random.rand(1,9)[0])
        labels.append(np.random.randint(2, size = 1)[0])
    return chars,second_chars,labels


config = {'algorithm': 'C4.5'}

chars,second,labels = return_amount_of_samples(50)

df = pd.DataFrame(list(zip(chars,second,labels)),columns=['first','second','Decision'])


model = chef.fit(df, config)

print(df)







