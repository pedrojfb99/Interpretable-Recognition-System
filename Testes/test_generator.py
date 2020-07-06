from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint
from numpy.random import random as rd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import load_model
from collections import Counter
from numba import cuda
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm
import cv2 as cv
import datetime
import tensorflow
import random
from itertools import islice



def generate_fake_characteristics(n):

    full = []

    range_of_each_sample = [
            [0,1,2],
            [0,1,2,3],
            [0,1,2],
            [1,2,3],
            [1,2],
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4],
            [0,1,2]

        ]

    for i in range(n):
        fake_chars = []
        for r in range_of_each_sample:
            selected = random.choice(r)
            fake_chars.append(selected)

        full.append(fake_chars)



    return full



def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)


    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)

    chars = generate_fake_characteristics(n_samples)

    return [np.asarray(x_input), np.asarray(chars)]



# generate points in latent space
x_input,chars_input = generate_latent_points(50, 1)

g_model = load_model("..//generator_models//generator_model810.h5")

# predict outputs
X = g_model.predict([x_input, chars_input])

plt.imshow(X)
plt.show()