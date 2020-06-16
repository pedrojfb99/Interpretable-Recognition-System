import pickle
from tensorflow.keras.models import load_model
import os
from random import randrange
import time
#Importamos a funçãodo A estrela
from A_star import call_a_star

# Resultados : [(x,y,label),...,..]
results = []

# Definir lista de treino
train = []

# Definir lista de teste
test = []

discriminator = load_model(os.getcwd() + "//discriminator_model8100.h5")

images = pickle.load(open("needed_correct_images.p", "rb"))
ks = list(images.keys())
ks = ks[:8000]

def get_random_different_image(id):
    """[summary]

    Args:
        id ([id]): [Recebe o id que nao quer que repita]

    Returns:
        [new_index]: [Novo index nao repetido]
    """    

    new_index = id
    print(new_index)
    while(id == new_index):

     new_index = randrange(0,8000)


    return new_index


# Percorrer as imagens treino : 8000
for image in range(len(images)):
    


    atual = ks[image]
    # Prever 20 casos legitimos . Label : 1
    for i in range(20):

        start = time.time()

        # Gerar uma combinação para a imagem
        aux1 = call_a_star(atual,discriminator)
        print(time.time() - start)


        # Gerar segunda combinação
        aux2 = call_a_star(atual,discriminator)
        print(time.time() - start)


        exit()

        # Label
        label = 1

        train.append((aux1, aux2, label))


    print("Acabei as 20 verdadeiras, vou agora para as falsas")
    # Prever 60 casos falsos . Label : 0
    for i in range(60):

        # Gerar uma combinação para a imagem
        aux1= call_a_star(atual,discriminator)


        # Gerar segunda combinação falsa
        false = get_random_different_image(image)

        aux2= call_a_star(ks[false],discriminator)

        # Label
        label = 0

        train.append((aux1, aux2, label))
