import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNNSampler():
    def __init__(self, cnn_layers_interval = [0, 4], fc_layers_inverval = [2,3], kernel_sizes = [3,4,5], allowed_acivations = [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu], use_resnet = 0,
                 use_batchnorm = 0, use_dropout = 0, img_dim = [28, 28], nr_classes=10, apply_softmax = False, seed = 42, max_nr_pools = 2):

        self.cnn_layers_interval = cnn_layers_interval
        self.fc_layers_inverval = fc_layers_inverval
        self.kernel_sizes = kernel_sizes
        self.allowed_acivations = allowed_acivations
        self.use_resnet = use_resnet
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.img_dim = img_dim
        self.nr_classes = nr_classes
        self.max_nr_pools = max_nr_pools

        self.apply_softmax = apply_softmax
        np.random.seed(seed)


    def get_network(self, input_ph):
        current_handle = input_ph
        print("asdf")
        activation_fnc = np.random.choice(self.allowed_acivations)
        use_resnet_bool = self.use_resnet > np.random.randint(0, 100)/100.0
        use_batchnorm_bool = self.use_batchnorm> np.random.randint(0, 100)/100.0
        use_dropout_bool = self.use_dropout > np.random.randint(0, 100)/100.0

        nr_cnn_layers = np.random.randint(self.cnn_layers_interval[0], self.cnn_layers_interval[1])
        nr_fc_layers = np.random.randint(self.fc_layers_inverval[0], self.fc_layers_inverval[1])

        for i in range(nr_cnn_layers):
            # add a cnn layer

        for i in range(nr_fc_layers):
            # add a fully connected layer

            # maybe add dropout
        return  current_handle







if __name__ == "__main__":
    print("Test CNN Sampler")

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    sampler = CNNSampler()

    input_ = tf.placeholder('float32', shape=(None, 28, 28))
    input_x = tf.reshape(input_, [-1, 28, 28, 1])

    sampler.get_network(input_x)