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
    trainY = []
    ks = list(images_all.keys())
    for i in tqdm(islice(range(len(images_all)),8000)):
        im = cv.imread(find(ks[i], path_to_watch))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        resize = cv.resize(im, (256, 256), interpolation=cv.INTER_AREA)

        trainX.append(resize)
        trainY.append(images_all[ks[i]])


    return np.asarray(trainX), np.asarray(trainY)


#trainX,trainY  = get_training()
#pickle.dump(trainX,open(path_to_watch + "dataset.p","wb"))
#pickle.dump(trainY,open(path_to_watch + "yis.p","wb"))

#trainX = pickle.load(open("dataset.p","rb"))
#trainY = pickle.load(open("yis.p","rb"))


# Converts the range of the pixels to [-1,1]
def load_real_samples(trainX,trainY):
    # convert from unsigned ints to floats
    # scale from [0,255] to [-1,1]
    X = (trainX - 127.5) / 127.5
    return [X, trainY]


def get_array_images(ix):

    size = (192,256)
    aux_x = []
    aux_y = []
    ks = list(images_all.keys())
    for index in ix:
        im = cv.imread(find(ks[index], path_to_watch))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)


        normalized_image = np.zeros(size)
        normalized_image = cv.normalize(im, normalized_image, 0, 255, cv.NORM_MINMAX)
        resize = cv.resize(normalized_image, size, interpolation=cv.INTER_AREA)


        aux_x.append(resize)
        aux_y.append(images_all[ks[index]])

    aux_x = load_real_samples(np.asarray(aux_x),np.asarray(aux_y))
    return aux_x


# example of smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
    return y + rd(y.shape) * 0.1


# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
    return y - 0.3 + (rd(y.shape) * 0.2)



# select real samples
def generate_real_samples(n_samples):
    # choose random instances
    ix = randint(0, 8000, n_samples)
    X, chars = get_array_images(ix)


    y = ones((n_samples, 1))
    # smooth labels
    y = smooth_positive_labels(y)


    indices = np.random.choice(np.arange(y.size), replace=False, size=int(y.size * 0.2))
    y[indices] = 0


    return [X, chars], y


# We put the pixel values back to normal [0,255]
def normalize(X):
    # scale from [-1,1] to  [0,255]
    X = (X * 127.5) + 127.5
    return X.astype(int)




#Fake samples


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




# use the generator to generate n fake examples
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input,chars_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs
    X = g_model.predict([x_input, chars_input])


    y = zeros((n_samples,1))


    y = smooth_negative_labels(y)

    indices = np.random.choice(np.arange(y.size), replace=False, size=int(y.size * 0.2))
    y[indices] = 1

    return [X, chars_input], y




# Best practice initialiser for GANS
initialWeights = RandomNormal(mean=0.0, stddev=0.02, seed=None)



def define_generator(latent_dim,n_classes = 30):

        #We define the characteristics
        cts = Input(shape=(9, ))


        layer_car = Flatten()(Embedding(n_classes, 32)(cts))


        n_nodes = 4 * 4 * 3

        layer_car = Dense(n_nodes)(layer_car)


        layer_car = Reshape((4, 4, 3))(layer_car)


        in_lat = Input(shape=(latent_dim, ))

        n_nodes_2 = 512 * 4 * 4

        gen = Dense(n_nodes_2, activation='relu', input_dim=latent_dim)(in_lat)
        gen = Reshape((4, 4, 512))(gen)
        gen = Dropout(0.4)(gen)


        merge = concatenate([gen, layer_car])


        #Upsample to 16 * 12


        gen = UpSampling2D(size = (4, 3), data_format=None, interpolation='nearest')(gen)
        gen = Conv2D(1024, 6, activation='relu', padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)


        #Upsample to 32 * 24
        gen = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest',)(gen)
        gen = Conv2D(512, 6,activation='relu',  padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)


        # Upsample to 64*48
        gen = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(gen)
        gen = Conv2D(256, 6,activation='relu',  padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)


        # Upsample to 128*96
        gen = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(gen)
        gen = Conv2D(128, 6,activation='relu',  padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)


        # Upsample to 256*192
        gen = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(gen)
        gen = Conv2D(64, 6, activation='relu', padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)




        ### Add complexity to the model


        # Symetry layer
        gen = UpSampling2D(size=(1, 1), data_format=None, interpolation='nearest')(gen)
        gen = Conv2D(64, 6, activation='relu', padding='same',kernel_initializer=initialWeights)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)



        # output layer
        output_layer = Conv2D(3, (6, 6), activation='tanh', padding='same')(gen)


        generator_model = Model([in_lat, cts], output_layer)


        return generator_model




def define_discriminator(in_shape=(256, 192, 3),n_classes = 30):



    #we define the input shape of the characteristics
    cts = Input(shape=(9,))

    #We embed the characteristics, which mean that for each characteristics
    layer_car = Flatten()(Embedding(n_classes, 32)(cts))

    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]

    #We create the Dense layer with the number of nodes of the required image
    layer_car = Dense(n_nodes)(layer_car)

    #We reshape the layer so it has the same dimensions has the image required
    layer_car = Reshape((in_shape[0], in_shape[1], 3))(layer_car)


    #Create the input for the image
    input_image = Input(shape=in_shape)

    #Merge the image with the label
    merge = concatenate([input_image,layer_car])


    # Input - 256 * 192
    ds = Conv2D(64, (4, 4), padding='same', input_shape=in_shape, kernel_initializer=initialWeights)(merge)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)



    # Downsample the image to 128 * 96
    ds = Conv2D(128, (4, 4), strides = (2, 2), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)



    # Downsample the image to 64 * 48
    ds = Conv2D(256, (4, 4), strides = (2, 2), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)


    # Downsample the image to 32 * 24
    ds = Conv2D(512, (4, 4), strides = (2, 2), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)


    # Downsample the image to 16 * 12
    ds = Conv2D(1024, (4, 4), strides = (2, 2), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)




    # Downsample the image to 4 * 4
    ds = Conv2D(1024, (4, 4), strides = (4, 3), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)



    # Downsample the image to 4 * 4
    ds = Conv2D(1024, (4, 4), strides = (1, 1), padding='same',kernel_initializer=initialWeights)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)
    ds = Dropout(0.6)(ds)



    # Flatten to 1 simple layer
    ds = Flatten()(ds)
    output_layer = Dense(1, activation='sigmoid')(ds)


    #We define the model
    discriminator = Model([input_image, cts], output_layer)


    # compile model
    opt = Adam(lr=0.00002,beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return discriminator



def define_gan(discriminator, generator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    gen_noise,  gen_label = generator.input

    gen_output = generator.output

    gan_output = discriminator([gen_output, gen_label])


    gan = Model([gen_noise,gen_label],gan_output)

    # compile model
    opt = Adam(lr=0.00002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return gan




def summarize_performance(epoch, generator, discriminator, latent_dim, n_samples=100):
    # Prepare real samples
    X_real, y_real = generate_real_samples(n_samples)

    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)

    # Fake examples
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)

    #Print the summary
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    save_plot(X_fake, epoch)

    generator.save("generator_models//generator_model{}.h5".format(epoch + 1))
    discriminator.save("discriminator_models//discriminator_model{}.h5".format(epoch + 1))



def save_plot(data, epoch, n=7):
    aux = normalize(data[0])
    # plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(aux[i])

    plt.savefig("generated//generated_plot{}.png".format(epoch + 1))
    plt.close()



# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=32):


    bat_per_epo = 64
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in tqdm(range(n_epochs)):
        final_gan_loss = []
        final_d_loss = []
        final_g_loss = []
        # enumerate batches over the training set
        for j in range(bat_per_epo):

            #Train first time
            # get randomly selected 'real' samples
            [X_real,real_chars], y_real = generate_real_samples(half_batch)

            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real,real_chars], y_real)

            # generate 'fake' examples
            [X_fake,fake_chars], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)


            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake,fake_chars], y_fake)


            #Train a second time

            # get randomly selected 'real' samples
            [X_real, real_chars], y_real = generate_real_samples(half_batch)

            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, real_chars], y_real)

            # generate 'fake' examples
            [X_fake, fake_chars], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, fake_chars], y_fake)


            # prepare points in latent space as input for the generator
            [X_gan,chars_input] = generate_latent_points(latent_dim, half_batch)

            # create inverted labels for the fake samples
            y_gan = ones((half_batch, 1))


            # update the generator via the discriminator's error
            g_loss,_ = gan_model.train_on_batch([X_gan,chars_input], y_gan)

            # summarize loss on this batch

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

            final_d_loss.append(d_loss1)
            final_g_loss.append(d_loss2)
            final_gan_loss.append(g_loss)


        if (i + 1) % 100 == 0:

            summarize_performance(i, g_model, d_model,latent_dim)

            x = list(range(1, 65))

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.suptitle('Sharing x per column, y per row')
            ax1.title.set_text("Real Discriminator Loss")
            ax1.plot(x, final_d_loss)
            ax2.title.set_text("Fake Discriminator Loss")
            ax2.plot(x, final_g_loss, 'tab:orange')
            ax3.title.set_text("GAN Discriminator Loss")
            ax3.plot(x, final_gan_loss, 'tab:red')


            plt.savefig("losses//loss_epoch_{}".format(i + 1))


import tensorflow.keras.backend as K
cfg = tensorflow.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
tensorflow.compat.v1.Session(config=cfg)





latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator= define_generator(latent_dim)
# create the gan
gan = define_gan(discriminator, generator)
# load image data
# train model
train(generator, discriminator, gan, latent_dim)

