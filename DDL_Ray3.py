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
    def transfer_layer_remove_size():
        return np.random.randint(1, 2)

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

    random_model_size = 5
    transfer_model_size = 3
    predict_model_size = 3

    epochs = 10
    mini_epochs = 2
    early_stop_rate = 0.05 * 100
    data_train_size = 50000 - 49000


@ray.remote
class RandomModel(object):
    def make_new_model(self, input_shape, output_dim):
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

    def make_transfer_model(self, output_dim, transfer_model):
        transfer_model_copy = k.models.clone_model(transfer_model)
        transfer_model_copy.set_weights(transfer_model.get_weights())

        model_input = transfer_model_copy.input
        model_output = transfer_model_copy.layers[-C.transfer_layer_remove_size() - 1].output

        try:
            model_output = k.layers.Flatten()(model_output)
        except ValueError:
            pass

        for i in range(C.dense_size()):
            model_output = k.layers.Dense(
                units=C.unit_size(),
                activation=C.activation(),
                name=str(np.random.rand()),
            )(model_output)

            model_output = k.layers.BatchNormalization(name=str(np.random.rand()))(model_output)

        model_output = k.layers.Dense(units=output_dim, activation='softmax', name=str(np.random.rand()))(model_output)

        model = k.Model(inputs=model_input, outputs=model_output)
        model.compile(optimizer=C.optimizer(), loss=C.loss())

        return model

    def __init__(self, sample_data, transfer_model=None):
            (self.train_x, self.train_y), (self.validation_x, self.validation_y) = sample_data

            if transfer_model is None:
                self.model = self.make_new_model(self.train_x.shape[1:], self.train_y.shape[1])

            else:
                self.model = self.make_transfer_model(self.train_y.shape[1], transfer_model)

    def get_model(self):
        return self.model

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

    def make_and_train_new_models(self, model_size):
        random_models = [RandomModel.remote(self.data.get_random_sample())
                         for _ in range(model_size)]

        for random_model in random_models:
            random_model.train.remote()

        accuracies = [random_model.get_accuracy.remote() for random_model in random_models]
        self.accuracy_with_models += [(accuracy, model) for accuracy, model in zip(accuracies, random_models)]

    def make_and_train_transfer_models(self, model_size):
        self.sort_models()

        transfer_models = [RandomModel.remote(self.data.get_random_sample(), ray.get(model.get_model.remote()))
                           for accuracy, model in self.accuracy_with_models[:model_size]]

        for transfer_model in transfer_models:
            transfer_model.train.remote()

        accuracies = [transfer_model.get_accuracy.remote() for transfer_model in transfer_models]
        self.accuracy_with_models += [(accuracy, model) for accuracy, model in zip(accuracies, transfer_models)]

    def get_accuracy(self):
        self.sort_models()

        predict_models = []
        for i in range(C.predict_model_size):
            random_model = self.accuracy_with_models[i][1]
            predict_models.append(random_model)

        predicts = np.zeros(self.data.test_y.shape)

        for predict_model in predict_models:
            predicts += ray.get(predict_model.predict.remote(self.data.test_x))

        correct = 0
        for predict, test in zip(predicts, self.data.test_y):
            if np.argmax(predict) == np.argmax(test):
                correct += 1

        return correct / len(predicts)

    def sort_models(self):
        temp_accuracy_with_models = []
        for accuracy, model in self.accuracy_with_models:
            try:
                temp_accuracy_with_models.append((ray.get(accuracy), model))
            except TypeError:
                temp_accuracy_with_models.append((accuracy, model))

        self.accuracy_with_models = temp_accuracy_with_models
        self.accuracy_with_models.sort()
        self.accuracy_with_models.reverse()

    def plot_models(self):
        self.sort_models()
        for i in range(len(self.accuracy_with_models)):
            k.utils.plot_model(ray.get(self.accuracy_with_models[i][1].get_model.remote()),
                               to_file='./models/model%d accuracy %f.png' % (i, self.accuracy_with_models[i][0]),
                               show_shapes=True)


def main():
    ensemble_model = EnsembleModel()
    ensemble_model.make_and_train_new_models(C.random_model_size)
    ensemble_model.make_and_train_transfer_models(C.transfer_model_size)
    for accuracy, model in ensemble_model.accuracy_with_models:
        print('model accuracy : ', ray.get(model.get_accuracy.remote((ensemble_model.data.test_x, ensemble_model.data.test_y))))

    print('ensemble accuracy : ', ensemble_model.get_accuracy())
    ensemble_model.plot_models()


main()
