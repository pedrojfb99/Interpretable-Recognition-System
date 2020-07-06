# Implementation of A*
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from tqdm import tqdm
from tensorflow.keras.models import load_model
import os
from dataclasses import dataclass
import random
import tensorflow
import time


# All combs


aux = [
    [0, 1, 2],
    [0, 1, 2, 3],
    [0, 1, 2],
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [0, 1, 2]

]


@dataclass
class node:
    f: float
    g: float
    h: float
    comb: list
    parent: list
    prob: float


# Definir funções

media_dos_valores = 0.80




def calculate_prob(comb,X,discriminator):
    X_train, chars = [np.asarray([X]), np.asarray([comb])]

    next_prob = discriminator.predict([X_train, chars])[0][0]

    return next_prob


# G -> Custo de um movimento de uma combinação para outra

def g(current, next_comb,X,discriminator):
    # Current prob
    X_train, chars = [np.asarray([X]), np.asarray([current])]

    current_prob = discriminator.predict([X_train, chars])[0][0]

    # Next prob

    X_train, chars = [np.asarray([X]), np.asarray([next_comb])]

    next_prob = discriminator.predict([X_train, chars])[0][0]

    return current_prob - next_prob


# H -> A estimação do atual para a final
def h(comb,X,discriminator):
    X_train, chars = [np.asarray([X]), np.asarray([comb])]
    actual_probability = discriminator.predict([X_train, chars])[0][0]

    return media_dos_valores - actual_probability


# F -> Sum of the two above


def f(h_value, g_value):
    return h_value + g_value


def get_sucessors(comb, fixed):
    sucessors = []

    for el in range(len(aux)):

        if fixed[el] == 0:

            secondary = comb.copy()

            if comb[el] + 1 in aux[el]:
                secondary[el] = secondary[el] + 1
                sucessors.append(secondary)

            secondary = comb.copy()

            if comb[el] - 1 in aux[el]:
                secondary[el] = secondary[el] - 1

                sucessors.append(secondary)

    return sucessors


def get_random_combination_start():
    combination_random = []
    for i in aux:
        combination_random.append(random.choice(i))

    return combination_random


def create_node(comb, parent, h_v, g_v, f_v, x):


    return node(f_v, g_v, h_v, comb, parent, x)

aux2 = pickle.load(open(os.getcwd() + "//needed_correct_images.p", "rb"))
ks = list(aux2.keys())

try:
    probs_for_each_image = pickle.load(open("all_probabilities_estrela.p", "rb"))
except:
    probs_for_each_image = []


ks = ks[8000 + len(probs_for_each_image):]

# Find the image a given directory
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


path_to_watch = os.getcwd() + "//..//annotations//"

def call_a_star(image,discriminator):
    


    start_1 = time.time()

    minimo = 0
    to_save = []


    size = (192, 256)

    im = cv.imread(find(image, os.getcwd() + "//annotations//"))
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    normalized_image = np.zeros(size)
    normalized_image = cv.normalize(im, normalized_image, 0, 255, cv.NORM_MINMAX)
    resize = cv.resize(normalized_image, size, interpolation=cv.INTER_AREA)

    X = (np.asarray(resize) - 127.5) / 127.5
    actual_combination = aux2[image]

    X_train, chars = [np.asarray([X]), np.asarray([actual_combination])]

    actual_probability = discriminator.predict([X_train, chars])[0][0]



    # we initializae the open and close the list
    open_list = []
    closed_list = []

    start_comb = get_random_combination_start()

    open_list.append(create_node(start_comb, [], h(start_comb,X,discriminator), 0, 0, 0))
    counter = 0
    winner = None
    to_stay = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    while (True):

        maxim = 999
        q = None
        auxiliar = 0
        guardador = 0

        finished = False
        control = False
        control2 = False

        # Find lowest f in open_list
        for el in open_list:
            if el.f < maxim:
                maxim = el.f
                guardador = auxiliar
            auxiliar += 1

        q = open_list.pop(guardador)

        q_sucessors = get_sucessors(q.comb, to_stay)

        for sucessor in q_sucessors:


            # Calculamos o g , h e f do sucessor
            g_value = q.g + (g(q.comb, sucessor,X,discriminator))

            h_value = h(sucessor,X,discriminator)

            f_value = f(g_value, h_value)

            actual = create_node(sucessor, q.comb, h_value, g_value, f_value, calculate_prob(sucessor,X,discriminator))

            # Caso seja o nodo destino
            if calculate_prob(sucessor,X,discriminator) >= media_dos_valores:
                finished = True

                minimo = actual.prob
                to_save = actual.comb



            # COrrer open list
            for o in open_list:
                if sucessor == o.comb:
                    if o.f < actual.f:
                        control = True

            if not control:
                # COrrer closed list
                for c in closed_list:
                    if sucessor == c.comb:
                        if c.f < actual.f:
                            control2 = True

            if not control2:
                open_list.append(actual)


        closed_list.append(q)


        if ((len(open_list) == 0) or finished):
            break

        counter += 1

        if counter == 6:
            open_list = []
            closed_list = []

            start_comb = get_random_combination_start()

            open_list.append(create_node(start_comb, [], h(start_comb,X,discriminator), 0, 0, 0))
            counter = 0



    for p in closed_list:
        if p.prob > minimo:
            min = p.prob
            to_save = p.comb

    print(time.time() - start_1)
    return to_save
