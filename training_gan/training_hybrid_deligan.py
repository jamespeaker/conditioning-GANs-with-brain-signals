from training_gans.models.hybrid_deligan_eeg import build_generator, build_discriminator, build_gan
from training_gans.utils import print_results
import keras.backend as K
import numpy as np
import pickle as pkl
from keras.models import load_model
from keras.optimizers import Adam


def train_GAN(batch_size, epochs, img_classifier, dataset, eeg_embs, d_optim, g_optim):
    """
    Function to train the GAN and show the performance as it trains.

    Function first instantiates and compiles the generator, discriminator, GAN
    and parameters such as the noise dimension. It then uses nested for-loops
    to train in batches. For each batch we see fake and real combinations of
    labels, embeddings, and images being created. These are used to train all
    models. At the end of each epoch the performance metrics are printed and
    we see what the generator is currently producing, given the EEG embeddings.

    input: batch_size: mini-batch size.
    input: epochs: number of epochs.
    input: img_classifier: pretrained classifier to use in the auxiliary feature
           of the discriminator.
    input: dataset: dataset used, in the form [labels, EEG embeddings, images].
    input: eeg_embs: a list of the 20 eeg embeddings, one for each class.
    input: d_optim: the optimizer used to train the discriminator.
    input: g_optim: the optimizer used to train the generator.
    """

    K.set_learning_phase(False)

    all_labels, all_eeg_embs, all_real_images = dataset

    input_noise_dim = 126
    feature_encoding_dim = 126
    tot_num_images = all_real_images.shape[0]
    print('tot images: ', tot_num_images)
    num_classes = 20

    d = build_discriminator((64, 64), img_classifier)
    d.trainable = True
    d.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=d_optim)

    g = build_generator(input_noise_dim, feature_encoding_dim)
    g.compile(loss='categorical_crossentropy', optimizer=g_optim)

    d_on_g = build_gan(input_noise_dim, feature_encoding_dim, g, d)
    d_on_g.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=g_optim)

    num_batches = int(tot_num_images / batch_size)

    print("Number of batches:", num_batches)

    for epoch in range(1, epochs + 1):
        for index in range(num_batches):
            # generate noise from a uniform distribution
            noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

            # get some random embedding and label pairings from the dataset (these are
            # used these to generate the fake images)
            rand_indices = np.random.randint(0, tot_num_images, batch_size)
            fake_labels = all_labels[rand_indices]
            fake_eeg_embs = all_eeg_embs[rand_indices]

            # get real images and corresponding labels and eeg embeddings
            real_labels_batch = all_labels[index * batch_size: (index + 1) * batch_size]
            real_images_batch = all_real_images[index * batch_size: (index + 1) * batch_size]

            # generate fake images using the generator
            generated_images = g.predict([noise, fake_eeg_embs], verbose=0)

            # train discriminator on real images (target is 1 - meaning real images)
            d_loss_real = d.train_on_batch(real_images_batch,
                                           [np.array([1] * batch_size), np.array(real_labels_batch)])
            # train discriminator on fake images (target is 0 - meaning fake images)
            d_loss_fake = d.train_on_batch(generated_images,
                                           [np.array([0] * batch_size),
                                            np.array(fake_labels).reshape(batch_size, num_classes)])
            d_loss = (d_loss_fake[0] + d_loss_real[0]) * 0.5

            d.trainable = False
            d.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=d_optim)

            # train the GAN. The generator will generate fake images (target is 1 - meaning real images)
            g_loss = d_on_g.train_on_batch([noise, fake_eeg_embs],
                                           [np.array([1] * batch_size),
                                            np.array(fake_labels).reshape(batch_size, num_classes)])
            d.trainable = True
            d.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=d_optim)

        print("Epoch {0}: discriminator_loss : {1:.4f}, generator_loss : {2:.4f}.".format(epoch + 1, d_loss, g_loss[0]))

        print_results(g, eeg_embs)


# load dataset
with open("../build_dataset/dataset_shuf.pkl", "rb") as f:
    dataset_shuf = pkl.load(f)

# load EEG embeddings
with open("../build_dataset/all_eeg_embeddings.pkl", "rb") as f:
    all_eeg_embeddings = pkl.load(f)

# train
train_GAN(
    batch_size=100,
    epochs=300,
    img_classifier=load_model('../training_image_classifier/image_classifier.h5'),
    dataset=dataset_shuf,
    eeg_embs=all_eeg_embeddings,
    d_optim=Adam(lr=5e-5, beta_1=0.5),
    g_optim=Adam(lr=1e-3)
)