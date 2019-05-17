import keras as k
import numpy as np
import ray

print('redis_address : ')
redis_address = input()
ray.init(redis_address=redis_address)


def one_hot(x, dim):
    y = np.zeros([len(x), dim])
    for i in range(x.shape[0]):
        y[i][x[i][0]] = 1

    return y


class C(object):
    @staticmethod
    def conv2D_size():
        return np.random.randint(0, 10 - 9)

    @staticmethod
    def dense_size():
        return np.random.randint(0, 10 - 3)

    @staticmethod
    def filter_size():
        return np.random.randint(3, 20)

    @staticmethod
    def kernel_size():
        return np.random.randint(3, 30), np.random.randint(3, 30)

    @staticmethod
    def unit_size():
        return np.random.randint(3, 30 - 10)

    @staticmethod
    def activation():
        return np.random.choice(['relu', 'sigmoid', 'softplus', 'tanh', 'softsign'])

    @staticmethod
    def optimizer():
        optimizer = k.optimizers.get(np.random.choice(['Adam', 'SGD', 'RMSprop']))
        k.backend.set_value(optimizer.lr, 10 ** np.random.randint(-3, -1))
        return optimizer

    @staticmethod
    def loss():
        return np.random.choice(['categorical_crossentropy'])

    random_model_size = 10
    predict_model_size = 5

    epochs = 10
    mini_epochs = 2
    early_stop_rate = 0.05 * 100
    data_train_size = 50000 - 49000


@ray.remote
class RandomModel(object):
    def make_model(self, input_shape, output_dim):
        model_input = model_output = k.Input(shape=input_shape)

        for _ in range(C.conv2D_size()):
            model_output = k.layers.Conv2D(
                filters=C.filter_size(),
                kernel_size=C.kernel_size(),
                activation=C.activation(),
                padding='same',
            )(model_output)
            model_output = k.layers.BatchNormalization()(model_output)

        model_output = k.layers.Flatten()(model_output)

        for _ in range(C.dense_size()):
            model_output = k.layers.Dense(
                units=C.unit_size(),
                activation=C.activation(),
            )(model_output)
            model_output = k.layers.BatchNormalization()(model_output)

        model_output = k.layers.Dense(units=output_dim, activation='softmax')(model_output)

        model = k.Model(inputs=model_input, outputs=model_output)
        model.compile(optimizer=C.optimizer(), loss=C.loss())

        return model

    def __init__(self, sample_data):
        (self.train_x, self.train_y), (self.validation_x, self.validation_y) = sample_data
        self.model = self.make_model(self.train_x.shape[1:], self.train_y.shape[1])

    def train(self):
        for i in range(int(C.epochs / C.mini_epochs)):
            train_log = self.model.fit(self.train_x, self.train_y, epochs=C.mini_epochs, verbose=0)

            print(i, train_log.history['loss'])
            if 1 - (train_log.history['loss'][-1] / train_log.history['loss'][0]) < C.early_stop_rate:
                break

    def predict(self, x):
        return self.model.predict(x)

    def get_accuracy(self, validation_data=None):
        if validation_data is None:
            validation_x, validation_y = self.validation_x, self.validation_y
        else:
            validation_x, validation_y = validation_data

        prediction_y = self.predict(validation_x)

        correct = 0
        for validation, prediction in zip(validation_y, prediction_y):
            if np.argmax(validation) == np.argmax(prediction):
                correct += 1

        return correct / len(prediction_y)


class Data(object):
    def __init__(self):
        (train_x, train_y), (test_x, test_y) = k.datasets.mnist.load_data()

        self.train_x = np.stack([train_x] * 3, axis=3)
        self.train_y = one_hot(train_y.reshape([-1, 1]), 10)
        self.test_x = np.stack([test_x] * 3, axis=3)
        self.test_y = one_hot(test_y.reshape([-1, 1]), 10)

    def get_random_sample(self):
        indexes = np.random.permutation(len(self.train_x))

        return (self.train_x[indexes][:C.data_train_size], self.train_y[indexes][:C.data_train_size]), \
               (self.train_x[indexes][C.data_train_size:], self.train_y[indexes][C.data_train_size:])


class EnsembleModel(object):
    def __init__(self):
        self.data = Data()
        self.accuracy_with_models = []

    def train_new_model(self):
        random_models = [RandomModel.remote(self.data.get_random_sample())
                         for _ in range(C.random_model_size)]

        ray.get([random_model.train.remote() for random_model in random_models])
        accuracies = ray.get([random_model.get_accuracy.remote() for random_model in random_models])

        self.accuracy_with_models += [(accuracy, model) for accuracy, model in zip(accuracies, random_models)]
        self.accuracy_with_models.sort()
        self.accuracy_with_models.reverse()

    def get_accuracy(self):
        predict_models = []
        for i in range(C.predict_model_size):
            random_model = self.accuracy_with_models[i][1]
            predict_models.append(random_model)

        predicts = np.zeros(self.data.test_y.shape)
        for i in range(len(predict_models)):
            predicts += ray.get(predict_models[i].predict.remote(self.data.test_x))
            #k.utils.plot_model(ray.get(predict_models[i]).model, to_file='./models/model%d.jpg' % i, show_shapes=True)

        correct = 0
        for predict, test in zip(predicts, self.data.test_y):
            if np.argmax(predict) == np.argmax(test):
                correct += 1

        return correct / len(predicts)


def main():
    ensemble_model = EnsembleModel()
    ensemble_model.train_new_model()
    for accuracy, model in ensemble_model.accuracy_with_models:
        print('model accuracy : ', ray.get(model.get_accuracy.remote((ensemble_model.data.test_x, ensemble_model.data.test_y))))

    print('ensemble accuracy : ', ensemble_model.get_accuracy())


main()
