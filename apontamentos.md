

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



## LeakyReLU vs ReLU





Combining ReLU, the hyper-parameterized1 leaky variant, and variant with dynamic parametrization during learning confuses two distinct things:

- The comparison between ReLU with the leaky variant is closely  related to whether there is a need, in the particular ML case at hand,  to avoid saturation — Saturation is thee loss of signal to either zero  gradient2 or the dominance of chaotic noise arising from digital rounding3.
- The comparison between training-dynamic activation (called *parametric* in the literature) and training-static activation must be based on  whether the non-linear or non-smooth characteristics of activation have  any value related to the rate of convergence4.

The reason ReLU is never parametric is that to make it so would be  redundant.  In the negative domain, it is the constant zero.  In the  non-negative domain, its derivative is constant.  Since the activation  input vector is already attenuated with a vector-matrix product (where  the matrix, cube, or hyper-cube contains the attenuation parameters)  there is no useful purpose in adding a parameter to vary the constant  derivative for the non-negative domain.

When there is curvature in the activation, it is no longer true that  all the coefficients of activation are redundant as parameters.  Their  values may considerably alter the training process and thus the speed  and reliability of convergence.

For substantially deep networks, the redundancy reemerges, and there  is evidence of this, both in theory and practice in the literature.

- In algebraic terms, the disparity between ReLU and parametrically  dynamic activations derived from it approaches zero as the depth (in  number of layers) approaches infinity.
- In descriptive terms, ReLU can accurately approximate functions with curvature5 if given a sufficient number of layers to do so.

That is why the ELU variety, which is advantageous for averting the  saturation issues mentioned above for shallower networks is not used for deeper ones.

So one must decided two things.

- Whether parametric activation is helpful is often based on  experimentation with several samples from a statistical population.  But there is no need to experiment at all with it if the layer depth is  high.
- Whether the leaky variant is of value has much to do with the  numerical ranges encountered during back propagation.  If the gradient  becomes vanishingly small during back propagation at any point during  training, a constant portion of the activation curve may be problematic.  In such a scase one of the smooth functions or leaky RelU with it's  two non-zero slopes may provide adequate solution.

In summary, the choice is never a choice of convenience.







# TODO

- Explicar iomplementação dos modelos (discriminador / gerador) 
- Fazer diagrama CGAN !
- Experiencias c45 e MLP ! 
- RELU vs LeakyRELU
- ADAM vs RMSDrop
- Introducao
- Acabar artiifiaal netowrks teorica