from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import pickle
import matplotlib.pyplot as plt

from keras.layers import Flatten
from keras.layers import Embedding

images = pickle.load(open("needed_correct_images.p","rb"))
dataset = pickle.load(open("train_mlp_set.p","rb"))
test = pickle.load(open("test_mlp_set.p","rb"))



def return_amount_of_samples(n_samples):

    chars = []
    first_labels = []


    for n in range(n_samples):
        chars.append([np.random.rand(1,9),np.random.rand(1,9)])

        first_labels.append(np.random.randint(2, size = 1))



    return chars,first_labels

def define_mlp(n_classes=9):
    # Primeiro input
    first_chars = Input(shape=(9,))
    first_dense = Flatten()(Embedding(n_classes, 32)(first_chars))
    first_dense = Dense(128,)(first_dense)

    # Segundo input
    second_chars = Input(shape=(9,))
    second_dense = Flatten()(Embedding(n_classes, 32)(second_chars))
    second_dense = Dense(128,)(second_dense)


    # Concatenamos o primeiro com o segundo
    concatenated_chars = concatenate([first_dense, second_dense])

    mlp_ayx = Dense(256, activation='relu')(concatenated_chars)

    mlp_ayx = Dense(128, activation='relu')(mlp_ayx)

    out_layer = Dense(1, activation='sigmoid')(mlp_ayx)

    model_mlp = Model([first_chars, second_chars], out_layer)

    model_mlp.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])

    return model_mlp




X_train = []
X_train_aux = []
Y_train = []


for x in dataset:
    X_train.append(np.asarray(x[0]))
    X_train_aux.append(np.asarray(x[1]))
    Y_train.append(x[2])



X_test = []
X_test_aux = []
Y_test = []


for x in dataset:
    X_test.append(np.asarray(x[0]))
    X_test_aux.append(np.asarray(x[1]))
    Y_test.append(x[2])

mlp = define_mlp()

history = mlp.fit(
    [X_train,X_train_aux],
    np.asarray(Y_train),
    batch_size=32,
    epochs=200,
    validation_data=([X_test,X_test_aux],np.asarray(Y_test) )

)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")


mlp.save("mlp_model.h5")
#X,y = return_amount_of_samples(100)




'''
ks = images.keys()
for i in ks:
    print("Image : {}   -  {} \n".format(i,images[i]))

'''