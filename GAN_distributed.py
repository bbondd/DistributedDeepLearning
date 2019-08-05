import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from keras.preprocessing import image

tf.enable_eager_execution()

noise_dimension = 128
(train_x, train_y), (_, _) = k.datasets.mnist.load_data()
train_x = train_x[:1000]
train_x = np.expand_dims(train_x, axis=3) / 255


class GAN(object):
    @staticmethod
    def make_discriminator():
        model_input = model_output = k.Input(shape=[28, 28, 1])
        model_output = k.layers.Conv2D(filters=32,
                                       kernel_size=[5, 5],
                                       strides=[2, 2],
                                       padding='same',
                                       )(model_output)
        model_output = k.layers.LeakyReLU()(model_output)
        model_output = k.layers.Conv2D(filters=64,
                                       kernel_size=[5, 5],
                                       strides=[2, 2],
                                       padding='same',
                                       )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.LeakyReLU()(model_output)
        model_output = k.layers.Conv2D(filters=128,
                                       kernel_size=[5, 5],
                                       strides=[2, 2],
                                       padding='same',
                                       )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.LeakyReLU()(model_output)

        model_output = k.layers.Flatten()(model_output)
        model_output = k.layers.Dense(units=1)(model_output)

        model = k.Model(inputs=model_input, outputs=model_output)
        return model

    @staticmethod
    def make_generator():
        model_output = model_input = k.Input(shape=[noise_dimension])
        model_output = k.layers.Dense(units=7 * 7 * 128)(model_output)
        model_output = k.layers.Reshape([7, 7, 128])(model_output)
        model_output = k.layers.Conv2DTranspose(filters=64,
                                                kernel_size=[5, 5],
                                                strides=[2, 2],
                                                padding='same',
                                                activation='relu'
                                                )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.Conv2DTranspose(filters=32,
                                                kernel_size=[5, 5],
                                                strides=[2, 2],
                                                padding='same',
                                                activation='relu'
                                                )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.Conv2DTranspose(filters=1,
                                                kernel_size=[5, 5],
                                                strides=[1, 1],
                                                padding='same',
                                                activation='tanh'
                                                )(model_output)
        model_output = k.layers.Lambda(lambda x: (x + 1) / 2)(model_output)

        return k.Model(inputs=model_input, outputs=model_output)

    def make_adversarial(self):
        model = k.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)

        return model

    def __init__(self):
        self.discriminator = self.make_discriminator()
        self.generator = self.make_generator()
        self.adversarial = self.make_adversarial()

    def train_discriminator(self):
        noise = np.random.uniform(-1, 1, [len(train_x), noise_dimension])
        generated_array = self.generator.predict(noise)

        xs = np.concatenate([train_x, generated_array], axis=0).astype('float32')
        ys = np.concatenate([np.ones([len(train_x), 1]), np.zeros([len(generated_array), 1])], axis=0)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(xs)
            loss = tf.losses.sigmoid_cross_entropy(ys, predictions)
        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return loss

    def train_generator(self):
        xs = np.random.uniform(-1, 1, [len(train_x), noise_dimension]).astype('float32')
        ys = np.ones([len(xs), 1])

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

        with tf.GradientTape() as tape:
            predictions = self.adversarial(xs)
            loss = tf.losses.sigmoid_cross_entropy(ys, predictions)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return loss

    def generate_image(self, image_size):
        noise = np.random.uniform(-1, 1, [image_size, noise_dimension])
        image_arrays = self.generator.predict(x=noise) * 255

        for i in range(image_size):
            image.save_img(x=image_arrays[i], path='./images/%d.jpg' % i)


def main():
    gan = GAN()

    k.utils.plot_model(gan.discriminator, to_file='discriminator.png', show_shapes=True)
    k.utils.plot_model(gan.generator, to_file='generator.png', show_shapes=True)

    for i in range(10000):
        discriminator_loss = gan.train_discriminator()
        generator_loss = gan.train_generator()

        if i % 100 == 0:
            print('generator loss : ', generator_loss)
            print('discriminator loss : ', discriminator_loss)

            print(i)
            gan.generate_image(10)


main()

