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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm
import cv2 as cv
import datetime
import tensorflow
import random

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
    for i in tqdm(range(len(images_all))):
        im = cv.imread(find(ks[i], path_to_watch))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        resize = cv.resize(im, (128, 128), interpolation=cv.INTER_AREA)

        trainX.append(resize)
        trainY.append(images_all[ks[i]])


    return np.asarray(trainX), np.asarray(trainY)


trainX,trainY  = get_training()
pickle.dump(trainX,open(path_to_watch + "dataset.p","wb"))
pickle.dump(trainY,open(path_to_watch + "yis.p","wb"))

#trainX = pickle.load(open("dataset.p","rb"))
#trainY = pickle.load(open("yis.p","rb"))


# Converts the range of the pixels to [-1,1]
def load_real_samples():
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5

    return [X, trainY]


# select real samples
def generate_real_samples(dataset, n_samples):
    images, chars = dataset

    # choose random instances
    ix = randint(0, dataset[0].shape[0], n_samples)


    # retrieve selected images
    X, chars = images[ix], chars[ix]

    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return [X, chars], y


# We put the pixel values back to normal [0,255]
def normalize(X ):
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


    # We give them the label
    y = zeros((n_samples, 1))
    return [X, chars_input], y




def define_generator(latent_dim,n_classes = 30):

    #We define the characteristics
    cts = Input(shape=(9, ))


    layer_car = Flatten()(Embedding(n_classes, 64)(cts))


    n_nodes = 4 * 4 * 3

    layer_car = Dense(n_nodes)(layer_car)


    layer_car = Reshape((4, 4, 3))(layer_car)


    in_lat = Input(shape=(latent_dim, ))

    n_nodes_2 = 256 * 4 * 4

    gen = Dense(n_nodes_2, input_dim=latent_dim)(in_lat)

    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((4, 4, 256))(gen)



    merge = concatenate([gen, layer_car])


    # Upsample from 4*4 to 8*8
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Upsample from 8*8 to 16*16
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Upsample from from 16*16 to 32 *32
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Upsample from from 32*32 to 64 *64
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Upsample from from 64*64 to 128*128
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # output layer
    output_layer = Conv2D(3, (3, 3), activation='tanh', padding='same')(gen)

    generator = Model([in_lat, cts], output_layer)


    return generator






def define_discriminator(in_shape=(128, 128, 3),n_classes = 30):

    #we define the input shape of the characteristics
    cts = Input(shape=(9,))

    #We embed the characteristics, which mean that for each characteristics
    layer_car = Flatten()(Embedding(n_classes, 64)(cts))

    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]

    #We create the Dense layer with the number of nodes of the required image
    layer_car = Dense(n_nodes)(layer_car)

    #We reshape the layer so it has the same dimensions has the image required
    layer_car = Reshape((in_shape[0], in_shape[1], 3))(layer_car)

    #Create the input for the image
    input_image = Input(shape=in_shape)

    #Merge the image with the label
    merge = concatenate([input_image,layer_car])


    # Input - 128*128
    ds = Conv2D(64, (3, 3), padding='same', input_shape=in_shape)(merge)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Downsample the image to 64*64
    ds = Conv2D(96, (3, 3), padding='same', input_shape=in_shape)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Downsample the image to 32 * 32
    ds = Conv2D(128, (3, 3), padding='same', input_shape=in_shape)(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Downsample the image to 16 * 16
    ds = Conv2D(128, (3, 3), strides = (2, 2), padding='same')(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Downsample the image to 8 * 8
    ds = Conv2D(256, (3, 3), strides = (2, 2), padding='same')(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Downsample the image to 4 * 4
    ds = Conv2D(512, (3, 3), strides = (2, 2), padding='same')(ds)
    ds = LeakyReLU(alpha=0.2)(ds)

    # Flatten to 1 simple layer
    ds = Flatten()(ds)
    ds = Dropout(0.4)(ds)
    output_layer = Dense(1, activation='sigmoid')(ds)


    #We define the model
    discriminator = Model([input_image, cts], output_layer)


    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
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
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt)

    return gan




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
    generator.save("discriminator_models//discriminator_model{}.h5".format(epoch + 1))



def save_plot(data, epoch, n=10):
    aux = normalize(data[0])
    # plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(aux[i])

    plt.savefig("generated//generated_plot{}.png".format(epoch + 1))
    plt.close()



# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=30000, n_batch=128):
    bat_per_epo = 64
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real,real_chars], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real,real_chars], y_real)

            # generate 'fake' examples
            [X_fake,fake_chars], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake,fake_chars], y_fake)
            # prepare points in latent space as input for the generator
            [X_gan,chars_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([X_gan,chars_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        if (i + 1) % 20 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)



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
