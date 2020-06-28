#Apontamentos

##TODOs

Switch from leakyRelu to relu on generator





## Upsampling vs Transposed Convolutional (to test)

When using the UpSampling2D  I get smoother and more natural images by the generator
 
With Transposed Convolutional the images like are more gridy(grid).

## Noise size 100 * 128 (to-test)

128 produces better results


## Batch size 64 vs 16 vs 8 (to-test)

Report says that a smaller batch size produces  better results.


## Loss functions  - binary_crossentropy vs Wasserstein vs RaLS (to-test)



## Over training




## Mode collapse(testing)

 
There might be a problem where the generated images have lower distribution, changing the learning rate of both
discriminator and generator helps this.


LR1 = 0.0002
LR2 = 0.0001


## Optimizer Adam vs Gradient Descent vs RMSDROP



## Images resolution

1st- > 128 * 128

2nd -> 512 * 512



## LeakyReLU vs ReLU

### LeakyReLU on discriminator e LeakyReLU on generator



### LeakyReLU on discriminator and ReLU on generator
