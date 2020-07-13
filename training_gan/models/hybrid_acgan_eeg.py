from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.models import Model



def build_generator(noise_dim, eeg_dim):
    """
    Builds the generator for the hybrid ACGAN using EEG embeddings.

    This generator is fairly standard as the discriminator is what makes the GAN
    a "hybrid ACGAN". It uses transposed convolutional layers to upsample to a
    64x63x3 image. The tanh layer at the end is used to transform the data into
    [-1,1] which is suitable for images.

    input: noise_dim: dimension of the noise.
    input: eeg_dim: dimension of the EEG embeddings.
    return: a model that takes noise and embeddings as input and outputs a 64x63x3
            image.
    """

    noise_input = keras.layers.Input(shape=(noise_dim,))
    eeg_input = keras.layers.Input(shape=(eeg_dim,))

    eeg_soft_max = Activation('softmax')(eeg_input)
    concat = Concatenate()([noise_input,eeg_soft_max])

    x = BatchNormalization(momentum=0.8)(concat)
    x = Dense(512*4*4, activation="relu")(x)
    x = Reshape((4, 4, 512))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same')(x)
    output = Activation('tanh')(x)

    return Model(inputs=[noise_input, eeg_input], outputs=[output])


def build_discriminator(classifier_mod, img_shape=(64, 64, 3)):
    """
    Builds the discriminator for the hybrid ACGAN using EEG embeddings.

    This disrciminator follows the 'hybrid' approach of the AC-GAN which
    involves using a pretrained classifier in the discriminator model.
    This classifier predicts the classification of the image. Therefore
    the discriminator has two outputs, one is for the real/fale prediction
    and one is for the image class. Here we are using Keras' functional API
    with the input being a 64x64x3 image. The model mostly uses the
    recommended sequence of layers suggested in the literature at the time.
    This includes repetitions of conv, ReLU, dropout and batch-norm.

    input: input_img_shape: shape of the input (the image)
    input: classifier_model: pretrained classifier (predicts the image class).
    return: a model with image input and two outputs: the real/fake prediction
            and the auxiliary prediction.
    """

    image_input = Input(shape=img_shape)
    x = Conv2D(16, (3, 3), strides=2)(image_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    flatten = Flatten()(x)

    # output for fake/not fake
    output = Dense(1, activation='sigmoid')(flatten)

    # output for type of class (not AGAN architecture though)
    classifier_mod.trainable = False
    aux = classifier_mod(inputs=[image_input])

    # Define the model
    model = keras.models.Model([image_input], [output, aux])
    return model


def build_gan(noise_dim, eeg_dim, gen_model, dis_model):
    """
    The GAN model. This is the composite of both the discriminator and
    generator.

    This function builds the composite model, allowing the generator to
    be trained in relation to the discriminator result (i.e. generator
    trained to trick the discriminator) while the discriminator is not in
    a trainable state.

    input: noise_dim: dimension of the noise.
    input: eeg_dim: dimension of the features (EEG embeddings).
    input: gen_model: the generator model.
    input: dis_model: the discriminator model.

    return: the composite model D(G(x,e)), for generator G, discriminator
            D, noise x and embeddings e.
    """

    noise_input = keras.layers.Input(shape=(noise_dim,))
    eeg_input = keras.layers.Input(shape=(eeg_dim,))

    gen_output = gen_model(inputs=[noise_input,eeg_input])

    # make discriminator not trainable
    dis_model.trainable = False

    # connect image output and eeg features input from generator as inputs to discriminator
    fake, aux = dis_model(inputs=[gen_output])

    # define gan model as taking latent noise and eeg_features and outputting a classification
    GAN_model = Model(inputs=[noise_input, eeg_input], outputs=[fake,aux])
    return GAN_model


