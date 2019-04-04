import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNNSampler():
    def __init__(self, input_dims, nr_classes, input, mode):


        self.nr_classes = nr_classes
        self.input_dims = input_dims
        self.networks_list = []

        # build networks
        with tf.name_scope("cnn_scope_1"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64 * 3])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_2"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=16,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_3"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            pool1_flat = tf.reshape(pool1, [-1, 7 * 7 * 32])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_4"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=128,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128 * 3])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_5"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64 ])
            dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_6"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            pool1_flat = tf.reshape(pool1, [-1, 7 * 7 * 32 * 3])
            dense = tf.layers.dense(inputs=pool1_flat, units=512, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_7"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                activation=None)

            input_flat = tf.reshape(conv1, [-1, 28*28])
            dense = tf.layers.dense(inputs=input_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=dropout, units=nr_classes)
        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_8"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense, units=nr_classes)

        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_9"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=16,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense, units=nr_classes)

        self.networks_list.append(logits)

        with tf.name_scope("cnn_scope_10"):
            conv1 = tf.layers.conv2d(
                inputs=input,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense, units=nr_classes)
        self.networks_list.append(logits)



    def get_network(self, id, sess):
        cnn_initializer = [var.initializer for var in tf.global_variables() if "cnn_scope_" + str(0) in var.name]
        sess.run(cnn_initializer)
        return self.networks_list[id]

    def nr_networks(self):
        return len(self.networks_list)





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

