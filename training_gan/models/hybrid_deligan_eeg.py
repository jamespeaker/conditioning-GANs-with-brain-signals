from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Reshape, LeakyReLU, Dropout, Input, multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.layers.core import Activation
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
import keras.backend as K


class DeLiGANLayer(Layer):
    """
    Layer based on the DeLiGAN.

    This layer allows the model to reparameterize the latent space as a (Gaussian)
    mixture model.
    """

    def __init__(self,
                 kernel_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """
        Initialise the the layer.

        Here the initialisers and regulisars (to build the weights) are set.
        """

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DeLiGANLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        """
        Build the weights of the layer (__call__() automatically runs this).

        Allows for the lazy creation of weights once the input shape is known. This
        method builds std and mean that will be used for f(x)=x*std+mean.

        input: input_shape: the input shape to the layer.
        """

        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.std = self.add_weight(shape=(input_dim,),
                                   name='std',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer)

        self.mean = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='mean')

        self.built = True

    def call(self, inputs):
        """
        The layers forward pass.

        input: inputs: input to the layer.
        output: output of the layer is f(input)=input*std+mean.
        """

        output = inputs * self.std
        output = K.bias_add(output, self.mean)
        return output

    def compute_output_shape(self, input_shape):
        """
        Computes the shape of the  layer output.
        """

        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)



def build_generator(noise_dim, feature_dim):
    """
    Builds the generator for the hybrid DeLiGAN using EEG embeddings.

    This generator uses the DeLiGAN approach of reparameterizing the latent space
    and then uses transposed convolutional layers to upsample to a 64x63x3 image.
    It is also a hybrid of the DeLiGAN as we use the auxiliary classifier GAN
    (AC-GAN) approach with a pretrained classifier (trained in
    image_classifier.ipynb).

    input: noise_dim: dimension of the noise.
    input: feature_dim: dimension of the EEG embeddings.
    return: a model that takes noise and embeddings as input and outputs a 64x63x3
            image.
    """

    noise_input = Input(shape=(noise_dim,))
    eeg_embedding_input = Input(shape=(feature_dim,))

    # softmax the EEG embedding
    eeg_soft_max = Activation('softmax')(eeg_embedding_input)

    # mixture model
    x = DeLiGANLayer(
        kernel_initializer=RandomUniform(minval=-0.2, maxval=0.2),
        bias_initializer=RandomUniform(minval=-1.0, maxval=1.0),
        kernel_regularizer=l2(0.01))(noise_input)

    # apply a transformation to the result of the mixture model
    x = Dense(feature_dim, activation="tanh")(x)
    x = multiply([x, eeg_soft_max])

    # standard generator
    x = BatchNormalization(momentum=0.8)(x)
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

    return Model(inputs=[noise_input, eeg_embedding_input], outputs=[output])


def build_discriminator(input_img_shape, classifier_model):
    """
    Builds the discriminator.

    This disrciminator follows the 'hybrid' approach of the AC-GAN but with
    a pretrained classifier (trained in image_classifier.ipynb). Here we
    are using Keras' functional API with the input being a 64x64x3 image
    and the output being the prediction of real/fake, as well as the
    auxiliary output which is a prediction of the image class.

    input: input_img_shape: shape of the input (the image)
    input: classifier_model: pretrained classifier (predicts the image class).
    return: a model with image input and two outputs: the real/fake prediction
            and the auxiliary prediction.
    """

    img_input = Input(shape=(input_img_shape[0], input_img_shape[1], 3))
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
    classifier_model.trainable = False

    # the pretrained classifier
    aux = classifier_model(inputs=[img_input])

    return Model(inputs=[img_input], outputs=[fake, aux])


def build_gan(noise_dim, feature_dim, g, d):
    """
    The GAN model. This is the composite of both the discriminator and
    generator.

    This function builds the composite model, allowing the generator to
    be trained in relation to the discriminator result (i.e. generator
    trained to trick the discriminator) while the discriminator is not in
    a trainable state.

    input: noise_dim: dimension of the noise.
    input: feature_dim: dimension of the features (EEG embeddings).
    input: g: the generator model.
    input: d: the discriminator model.

    return: the composite model D(G(x,e)), for generator G, discriminator
            D, noise x and embeddings e.
    """

    noise_input = Input(shape=(noise_dim,))
    eeg_embedding_input = Input(shape=(feature_dim,))
    g_output = g(inputs=[noise_input, eeg_embedding_input])
    # d.trainable = False
    fake, aux = d(inputs=[g_output])
    return Model(inputs=[noise_input, eeg_embedding_input], outputs=[fake, aux])