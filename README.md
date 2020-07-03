# Using EEG Signals to Condition Variants of GANs

MSc Computer Science thesis. The aim is to create a model capable of generating an approximation of the images seen by people. I attempt to replicate previous work (PeRCeiVe Lab) to condition variants of GANs with EEG brain signals. 

## Summary

This project aims to look at how deep learning can be applied to mind reading. It also gives insight on:
1. How the EEG brain signal embeddings compare to one-hot encoding, when used to condition a GAN.
2. What architectures of GANs are best suited for this problem.

<img src="project_visualization.png" width="600">

Figure 1. A visual representation of this project. The participant views a series of images and the EEG signals are encoded. These encodings are then used by the generator to attempt to predict the image seen.


## Encoder

The encoder seen in Figure 1 is a truncation of a pretrained model. This pretrained model was trained to classify 40 different image classes, using the EEG data. This is shown below in Figure 2. See that the encoder outputs a 126-length embedding of the EEG. The embedding length of 126  was chosen so we can reshape the embedding to the square cuboid shape of (3,3,14).

<img src="explaining_the_encoder.png" width="1000">

Figure 2. A visual representation of the classifier and how it is truncated to give the encoder.

<br />

**Result:** we now have an encoder which reduces the dimensionality of the EEG (on 128 channels) down to a 126-length embedding. We theorize that these embeddings may contain both visually-relevant and class-discriminative information extracted from the input signals.


## Generator

<ins> GAN Explanation </ins> <br />
A typical GAN is made up of the generator model and the disciminator model. The aim of the generator is to generate data mimicking the dataset. The aim of the discriminator is to classify the generated data as being from the real dataset or the fake dataset. The generator learns to trick the discriminator into wrongly classifying the fake dataset. If the adversarial process is successful, the generator can generate fake images indistinguishable from real images. 


<ins> Conditional GAN Explanation </ins> <br />
It becomes very challenging for GANs to generate images that are from different classes. To solve this, the generator and discriminator of the conditional GAN (CGAN) receieve information about the class of the image. Often the information is a one-hot encoding of the image class. This allows the generator to learn to generate images of the correct class. Below in Figure 3 we see how the EEG embedding is used in a CGAN.


<img src="explaining_cgan.png" width="1000">



Figure 3. A CGAN using convolutional layers in the discriminator and transposed convolutional layers in the generator. The EEG embedding is appended to the latent space of the generator and to the penultimate output of the discriminator.



I think the plan is to have:

(I should probs read the thing gary posted though about projects.)

1. an explanation of the other GAN architectures here

2. Then a results

3. Then a method section which refers to the code in repo.
i.e.
1. train EEG classifier
2. take out EEG embeddings
3. build image dataset
4. build full dataset
5. GAN models
6. training the GAN





