#Using the chefboost framework
#https://github.com/serengil/chefboost
from chefboost import Chefboost as chef

import numpy as np


import pandas as pd

import pickle


images = pickle.load(open("needed_correct_images.p","rb"))

dataset = pickle.load(open("train_mlp_set.p","rb"))

X_train = []
X_train_aux = []
Y_train = []

a = {

    'aux_0' : [],
    'aux_1': [],
    'aux_2': [],
    'aux_3': [],
    'aux_4': [],
    'aux_5': [],
    'aux_6': [],
    'aux_7': [],
    'aux_8': [],
    'aux_9': [],
    'aux_10': [],
    'aux_11': [],
    'aux_12': [],
    'aux_13': [],
    'aux_14': [],
    'aux_15': [],
    'aux_16': [],
    'aux_17': [],


     }

for x in dataset:
    contador = 0
    aux = []
    for i in x[0]:
        a['aux_{}'.format(contador)].append(i)
        contador += 1

    for i in x[1]:
        a['aux_{}'.format(contador)].append(i)
        contador += 1

    Y_train.append(x[2])


df = pd.DataFrame((zip(a['aux_0'], a['aux_1'], a['aux_2'], a['aux_3'], a['aux_4'], a['aux_5'], a['aux_6'],
                       a['aux_7'], a['aux_8'], a['aux_9'], a['aux_10'], a['aux_11'], a['aux_12'], a['aux_13'],
                       a['aux_14'], a['aux_15'], a['aux_16'], a['aux_17'], Y_train)),
                  columns=['first', 'second', 'third', 'fourth', 'fifht', '6', '7', '8', '9', '10', '11', '12',
                           '13', '14', '15', '16', '17', '18', 'Decision'])



if __name__ == '__main__':

    config = {'algorithm': 'C4.5', 'enableParallelism': True}
    model = chef.fit(df, config)
    fi = chef.feature_importance()
    print(fi)





