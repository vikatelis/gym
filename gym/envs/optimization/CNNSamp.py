import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNNSampler():
    def __init__(self, cnn_layers_interval = [0, 6], cnn_filter_interval = [16, 256], fc_layers_inverval = [2,3], fc_filter_inverval = [256, 1024], kernel_sizes = [3,4,5], allowed_acivations = [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu], use_resnet = 0,
                 use_batchnorm = 0, use_dropout = 0, dropout_interval=[10,50], img_dim = [28, 28], nr_classes=10, apply_softmax = False, seed = 42, max_nr_pools = 2):

        self.cnn_layers_interval = cnn_layers_interval
        self.cnn_filter_interval = cnn_filter_interval
        self.fc_layers_inverval = fc_layers_inverval
        self.fc_filter_inverval = fc_filter_inverval
        self.kernel_sizes = kernel_sizes
        self.allowed_acivations = allowed_acivations
        self.use_resnet = use_resnet
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_interval = dropout_interval
        self.img_dim = img_dim
        self.nr_classes = nr_classes
        self.max_nr_pools = max_nr_pools

        self.apply_softmax = apply_softmax
        np.random.seed(seed)


    def get_network(self, input_ph):
        current_handle = input_ph
        activation_fnc = np.random.choice(self.allowed_acivations)
        use_resnet_bool = self.use_resnet > np.random.randint(0, 100)/100.0
        use_batchnorm_bool = self.use_batchnorm> np.random.randint(0, 100)/100.0
        use_dropout_bool = self.use_dropout > np.random.randint(0, 100)/100.0

        dropout_rate = np.random.randint(*self.dropout_interval)/100.0

        nr_cnn_layers = np.random.randint(*self.cnn_layers_interval)
        nr_fc_layers = np.random.randint(*self.fc_layers_inverval)

        nr_pools = 0

        filter_sizes = np.sort(np.random.randint(*self.cnn_filter_interval, size=nr_cnn_layers))

        for i in range(nr_cnn_layers):
            k_size = np.random.choice(self.kernel_sizes)
            # add a cnn layer
            current_handle = tf.layers.conv2d(
                inputs=current_handle,
                filters=filter_sizes[i],
                kernel_size=[k_size, k_size],
                padding="same",
                activation=activation_fnc)
            # maybe batchnorm
            if use_batchnorm_bool:
                current_handle = tf.layers.batch_normalization(current_handle)

            # maybe pooling
            if min(self.img_dim)/max(2*nr_pools,1) > 7:
                current_handle = tf.layers.max_pooling2d(inputs=current_handle, pool_size=[2, 2], strides=2)
                nr_pools +=1


        # flatten
        current_handle = tf.reshape(current_handle, [-1, int(self.img_dim[0]/max(2*nr_pools,1) * self.img_dim[1]/max(2*nr_pools,1)
                                                     * filter_sizes[-1] * 3)])

        filter_sizes = np.sort(np.random.randint(*self.fc_filter_inverval, size=nr_fc_layers-1))[:: -1]

        for i in range(nr_fc_layers-1):
            # add a fully connected layer
            current_handle = tf.layers.dense(inputs=current_handle, units=filter_sizes[i], activation=tf.nn.relu)
            # maybe add dropout
            if use_dropout_bool:
                current_handle = tf.layers.dropout(inputs=current_handle, rate=dropout_rate, training= tf.estimator.ModeKeys.TRAIN)

        current_handle = tf.layers.dense(inputs=current_handle, units=self.nr_classes, activation=None)

        return current_handle






if __name__ == "__main__":
    print("Test CNN Sampler")

    sess = tf.Session()

    mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = np.eye(np.max(train_labels)+1)[train_labels]
    test_labels = np.eye(np.max(test_labels)+1)[test_labels]
    # one hot encode

    sampler = CNNSampler()

    input_ = tf.placeholder(tf.float32, shape=(None, 28, 28))
    input_x = tf.reshape(input_, [-1, 28, 28, 1])

    input_gt = tf.placeholder(tf.float32, shape=(None, 10))

    network = sampler.get_network(input_x)

    cur_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_gt, logits=network))


    # get optimizer
    var_list = [var for var in tf.trainable_variables()]
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.00002).minimize(network, var_list=var_list)


    # init vars
    sess.run(tf.global_variables_initializer())

    batch_size = 20
    n_epochs = 10
    show_loss_per_epoch = 10

    n_iter = int(n_epochs * train_images.shape[0]/batch_size)
    n_iter_show_loss = int(train_images.shape[0]/show_loss_per_epoch/batch_size)

    saved_losses = []
    # run for n epochs
    for iter in range(n_iter):
        # do update
        batch_low = (iter*batch_size)%train_images.shape[0]
        batch_high = ((iter+1)*batch_size)%train_images.shape[0]

        if batch_low>batch_high:
            continue # skip last inconplete batch

        feed_dict = {input_: train_images[batch_low:batch_high], input_gt: train_labels[batch_low:batch_high]}
        _, loss_fetch = sess.run([optim, cur_loss], feed_dict=feed_dict)
        # maybe print&store loss
        if iter % 1 == 0 or iter == 0:
            saved_losses.append(loss_fetch)
            print("loss at iteration "+ str(iter)+": "+str(loss_fetch))

