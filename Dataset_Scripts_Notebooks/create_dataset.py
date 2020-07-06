import pickle
from tensorflow.keras.models import load_model
import os
from random import randrange
import time
#Importamos a funçãodo A estrela
from A_star import call_a_star
import tensorflow
from tqdm import tqdm

import tensorflow.keras.backend as K
cfg = tensorflow.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
tensorflow.compat.v1.Session(config=cfg)


'''

# Resultados : [(x,y,label),...,..]
results = []

# Definir lista de treino
train = []

# Definir lista de teste
test = []

discriminator = load_model(os.getcwd() + "//discriminator_model8100.h5")

images = pickle.load(open("needed_correct_images.p", "rb"))
ks = list(images.keys())
ks = ks[8000:]

def get_random_different_image(id):
    """[summary]

    Args:
        id ([id]): [Recebe o id que nao quer que repita]

    Returns:
        [new_index]: [Novo index nao repetido]
    """    

    new_index = id
    while(id == new_index):

     new_index = randrange(0,2250)


    return new_index


# Percorrer as imagens treino : 8000
for image in tqdm(range(len(ks))):
    


    atual = ks[image]
    # Prever 5 casos legitimos . Label : 1
    for i in range(1):


        # Gerar uma combinação para a imagem
        aux1 = call_a_star(atual,discriminator)


        # Gerar segunda combinação
        aux2 = call_a_star(atual,discriminator)


        # Label
        label = 1

        train.append((aux1, aux2, label))


    # Prever 15 casos falsos . Label : 0
    for i in range(2):

        # Gerar uma combinação para a imagem
        aux1= call_a_star(atual,discriminator)


        # Gerar segunda combinação falsa
        false = get_random_different_image(image)

        aux2= call_a_star(ks[false],discriminator)

        # Label
        label = 0

        train.append((aux1, aux2, label))

    pickle.dump(train, open("test_mlp_set.p", "wb"))



'''

print(len(pickle.load(open("test_mlp_set.p","rb"))))