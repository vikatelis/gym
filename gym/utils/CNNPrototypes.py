import numpy as np
import tensorflow as tf
from tensorflow import keras


class CNNPrototypes():
    def __init__(self):
        self.networks_cache = {}

    def build_net(self,input_ph, id, nr_classes, mode,tag):
        # build networks
        if id == 1:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)

        elif id == 2:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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

                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)


        elif id == 3:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

                pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])
                dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)


        elif id == 4:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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

                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128 ])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)


        elif id == 5:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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


        elif id == 6:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

                pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])
                dense = tf.layers.dense(inputs=pool1_flat, units=512, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)


        elif id == 7:
            with tf.variable_scope("cnn_scope_"+tag):

                input_flat = tf.reshape(input_ph, [-1, 28*28])
                dense = tf.layers.dense(inputs=input_flat, units=1024, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                logits = tf.layers.dense(inputs=dropout, units=nr_classes)


        elif id == 8:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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


        elif id == 9:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
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

                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                logits = tf.layers.dense(inputs=dense, units=nr_classes)

        elif id == 10:
            with tf.variable_scope("cnn_scope_"+tag):
                conv1 = tf.layers.conv2d(
                    inputs=input_ph,
                    filters=64,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=128 ,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                logits = tf.layers.dense(inputs=dense, units=nr_classes)

        return logits


    def get_network(self,input_ph, input_ph_n, gt_ph,learn_rate, id, nr_classes, mode):

        tag = str(id)+"_"+str(nr_classes)+"_"+''.join([str(x)+"_" for x in input_ph.shape if not str(x) == "?"])
        if tag not in self.networks_cache.keys():
            self.networks_cache[tag] = self.build_net(input_ph,id,nr_classes,mode,tag)
            self.networks_cache[tag+"ph"] = input_ph
            self.networks_cache[tag + "ph_n"] = input_ph_n
            self.networks_cache[tag + "gt"] = gt_ph
            self.networks_cache[tag + "lr"] = learn_rate
            with tf.variable_scope("cnn_scope_" + tag):
                self.networks_cache[tag + "loss"] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gt_ph, tf.int32), logits=self.networks_cache[tag]))
                self.networks_cache[tag + "train_step"] = tf.train.GradientDescentOptimizer(self.networks_cache[tag + "lr"]).minimize(self.networks_cache[tag + "loss"])


        return self.networks_cache[tag], tag, self.networks_cache[tag+"ph"], self.networks_cache[tag + "gt"], \
               self.networks_cache[tag + "loss"], self.networks_cache[tag + "train_step"], self.networks_cache[tag + "lr"],  self.networks_cache[tag + "ph_n"]

    def nr_networks(self):
        return 10





if __name__ == "__main__":
    print("Test CNN Sampler")

    sess = tf.Session()

    mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = np.eye(np.max(train_labels)+1)[train_labels]
    test_labels = np.eye(np.max(test_labels)+1)[test_labels]
    # one hot encode

    sampler = CNNPrototypes()

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

