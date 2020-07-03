

## Upsampling vs Transposed Convolutional (to test)

When using the UpSampling2D  I get smoother and more natural images by the generator

With Transposed Convolutional the images like are more gridy(grid).

## Noise size 100 * 128 (to-test)

128 produces better results


## Batch size 64 vs 16 vs 8 (to-test)

Report says that a smaller batch size produces  better results.





## Dropout



Given that we know a bit about dropout, a question arises — why do we need  dropout at all? Why do we need to literally shut-down parts of a neural  networks?

***The answer to these questions is “to prevent over-fitting”.\***

A fully connected layer occupies most of the parameters, and hence,  neurons develop co-dependency amongst each other during training which  curbs the individual power of each neuron leading to over-fitting of  training data.

## Why do we use batch normalization?

We normalize the input layer by adjusting and scaling the activations. For example, when we have features from 0 to 1 and some from 1 to 1000, we  should normalize them to speed up learning. If the input layer is  benefiting from it, why not do the same thing also for the values in the hidden layers, that are changing all the time, and get 10 times or more improvement in the training speed.

Batch normalization reduces the amount by what the hidden unit values shift  around (covariance shift). To explain covariance shift, let’s have a  deep network on cat detection. We train our data on only black cats’  images. So, if we now try to apply this network to data with colored  cats, it is obvious; we’re not going to do well. The training set and  the prediction set are both cats’ images but they differ a little bit.  In other words, if an algorithm learned some X to Y mapping, and if the  distribution of X changes, then we might need to retrain the learning  algorithm by trying to align the distribution of X with the distribution of Y. ( Deeplearning.ai: Why Does Batch Norm Work? ([C2W3L06](https://www.youtube.com/watch?v=nUUqwaxLnWs)))





Also, batch normalization allows each layer of a network to learn by itself a little bit more independently of other layers.








## Mode collapse(testing)


There might be a problem where the generated images have lower distribution, changing the learning rate of both
discriminator and generator helps this.


LR1 = 0.0002
LR2 = 0.0001


## Optimizer Adam vs  RMSDROP



## Images resolution

1st- > 128 * 128

2nd -> 512 * 512





> \- A seção 2.2.1 não tem nada
>
> \- na seção 3.2.2 devias adicionar um par de ilustrações introdutórias das imagens utilizadas. Talvez uma imagem em grande, onde assinalas as regiões (pálpebras, pestanas, íris,…) de onde provêm os labels utilizados
>
> \- A resolução de algumas figuras está muito baixa, o que as torna quase ilegíveis. (Exemplo, figura 3.1)
>
> \- na equação 3.1 e nas restantes, tens que descrever todos os termos que aparecem. Por exemplo, falta descrever p(x)
>
> \- Na seção 4.2.1 está uma referência quebrada
>
> \- Na seção 4.4.1 a rede ainda deveria ser deixada mais tempo a treinar, porque ainda iria melhorar a taxa de acerto.
>
> 