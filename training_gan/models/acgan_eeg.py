from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.models import Model



def build_generator(noise_dim, eeg_dim):
    """
    Builds the generator for the ACGAN using EEG embeddings as conditioning.

    This generator is fairly standard as the discriminator is what makes the GAN
    an "ACGAN". It uses transposed convolutional layers to upsample to a
    64x63x3 image. The tanh layer at the end is used to transform the data into
    [-1,1] which is suitable for images.

    input: noise_dim: dimension of the noise.
    input: eeg_dim: dimension of the EEG embeddings.
    return: a model that takes noise and embeddings as input and outputs a 64x63x3
            image.
    """

    noise_input = keras.layers.Input(shape=(noise_dim,))
    eeg_input = keras.layers.Input(shape=(eeg_dim,))

    concat = Concatenate()([noise_input, eeg_input])

    x = BatchNormalization(momentum=0.8)(concat)
    x = Dense(512 * 4 * 4, activation="relu")(x)
    x = Reshape((4, 4, 512))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    output = Activation("tanh")(x)

    return Model(inputs=[noise_input, eeg_input], outputs=[output])


def build_discriminator(input_img_shape):
    """
    Builds the discriminator for the ACGAN using EEG embeddings as
    conditioning.

    This disrciminator follows the approach of the AC-GAN which uses the
    discriminator neural network to predict both the real/fake output and
    the auxiliary classification output (prediction of the image class).
    Here we are using Keras' functional API with the input being a 64x64x3
    image. The model mostly uses the recommended sequence of layers
    suggested in the literature at the time. This includes repetitions of
    conv, ReLU, dropout and batch-norm.

    input: input_img_shape: shape of the input (the image)
    return: a model with image input and two outputs: the real/fake prediction
            and the auxiliary prediction.
    """

    img_input = keras.layers.Input(shape=(input_img_shape[0], input_img_shape[1], 3))
    x = Conv2D(16, (3, 3), strides=2)(img_input)
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
    x = Flatten()(x)

    fake = Dense(1, activation='sigmoid')(x)
    aux = Dense(20, activation='softmax')(x)  # 20 for the number of classes
    return Model(inputs=[img_input], outputs=[fake, aux])


def build_gan(noise_dim, eeg_dim, gen_model, dis_model):
    """
    The GAN model. This is the composite of both the discriminator and
    generator.

    This function builds the composite model, allowing the generator to
    be trained in relation to the discriminator result (i.e. generator
    trained to trick the discriminator) while the discriminator is not in
    a trainable state.

    input: noise_dim: dimension of the noise.
    input: eeg_dim: dimension of the EEG embedding used as conditioning.
    input: gen_model: the generator model.
    input: dis_model: the discriminator model.

    return: the composite model D(G(x,e)), for generator G, discriminator
            D, noise x and embeddings e.
    """

    noise_input = keras.layers.Input(shape=(noise_dim,))
    eeg_input = keras.layers.Input(shape=(eeg_dim,))
    g_output = gen_model(inputs=[noise_input, eeg_input])
    dis_model.trainable = False
    fake, aux = dis_model(inputs=[g_output])
    return Model(inputs=[noise_input, eeg_input], outputs=[fake, aux])