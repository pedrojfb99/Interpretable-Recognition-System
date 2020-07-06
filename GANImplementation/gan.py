from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm
import cv2 as cv
import datetime
import tensorflow

# We define where we wanna search for the images
path_to_watch = os.getcwd() + "//annotations//"


# Loads the dictionary with the images
images_all = pickle.load(open("needed_correct_images.p", 'rb'))


# Find the image a given directory
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# Data Functions
def get_training():
    trainX = []
    ks = list(images_all.keys())
    for i in tqdm(range(8000)):
        im = cv.imread(find(ks[i], path_to_watch))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        resize = cv.resize(im, (32, 32), interpolation=cv.INTER_AREA)

        trainX.append(resize)
        if i == 8000:
            break

    return np.asarray(trainX)


#x  = get_training()
#pickle.dump(x,open(path_to_watch + "dataset.p","wb"))
x = pickle.load(open("dataset.p","rb"))


# Converts the range of the pixels to [-1,1]
def load_real_samples():
    # convert from unsigned ints to floats
    X = x.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5

    return X


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# We put the pixel values back to normal [0,255]
def normalize(X):
    # scale from [-1,1] to  [0,255]
    X = (X * 127.5) + 127.5
    return X.astype(int)


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs
    X = g_model.predict(x_input)

    # We give them the label
    y = zeros((n_samples, 1))
    return X, y


def define_generator(latent_dim):


    generator = Sequential()

    n_nodes = 256 * 4 * 4
    generator.add(Dense(n_nodes, input_dim=latent_dim))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Reshape((4, 4, 256)))

    # Upsample from 4*4 to 8*8
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))

    # Upsample from 8*8 to 16*16
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))

    # Upsample from from 16*16 to 32 *32
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))

    # output layer
    generator.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return generator




def define_discriminator(in_shape=(32, 32, 3)):



    discriminator = Sequential()

    # Input
    discriminator.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    discriminator.add(LeakyReLU(alpha=0.2))

    # Downsample the image to 16 * 16
    discriminator.add(Conv2D(128, (3, 3), strides = (2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))

    # Downsample the image to 8 * 8
    discriminator.add(Conv2D(256, (3, 3), strides = (2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))

    # Downsample the image to 4 * 4
    discriminator.add(Conv2D(512, (3, 3), strides = (2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))

    # Flatten to 1 simple layer
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return discriminator



def define_gan(discriminator, generator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    # connect them
    model = Sequential()

    # add generator
    model.add(generator)

    # add the discriminator
    model.add(discriminator)

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model




def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=150):
    # Prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)

    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)

    # Fake examples
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)

    # Print the summary
    print("Real Accuracy : {} %     || Fake Accuracy :  {} %  ".format(acc_real * 100, acc_fake * 100))

    save_plot(X_fake, epoch)

    generator.save("generator_models//generator_model{}.h5".format(epoch + 1))


def save_plot(data, epoch, n=10):
    aux = normalize(data)
    # plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(aux[i])

    plt.savefig("generated//generated_plot{}.png".format(epoch + 1))
    plt.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        if (i + 1) % 2 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)



# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(d_model, g_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

