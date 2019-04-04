import math
import random
from random import randint

import gym
from gym import spaces
import gym.spaces
from gym.utils import seeding
from gym.utils.cnn_standard import cnn_model_fn
from gym.utils.load_data import load_japanese_mnist
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

class SGDwithCNN(gym.Env):
    def __init__(self):
        # define dimensionality
        self.ndim = 1
        self.count = 0
        # range action
        self.min_action = -10*np.ones(self.ndim)
        self.max_action = 10*np.ones(self.ndim)
        # range observation - dimensionality
        self.low_state = np.append([-30.0],self.min_action)
        self.high_state = np.append([30.0],self.max_action)
        # range state - dimensionality
        self.low_state_p = -7*np.ones(self.ndim)
        self.high_state_p = 7*np.ones(self.ndim)
        # boxes
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=None, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=None, dtype=np.float32)
        self.hyper_space = spaces.Box(low=self.low_state_p, high=self.high_state_p, shape=None, dtype=np.float32)
        # init
        self.obs = self.observation_space.sample()
        self.state = self.hyper_space.sample()
        self.prev_loss = 0
        self.scale_function = 0
        # load dataset
        dataset = "FashionMNIST"
        if dataset == "MNIST":
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            self.X_train = X_train.astype(float) / 255.
            #self.Y_train = np.squeeze(y_train, axis=-1)
            self.Y_train = y_train
            # init placeholder
            self.input_ = tf.placeholder('float32',shape = (None,28,28))
            self.input_x = tf.reshape(self.input_, [-1, 28, 28, 1])
            self.input_y = tf.placeholder('float32',shape = (None))
        elif dataset == "CIFAR10":
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            self.X_train = X_train.astype(float) / 255.
            self.Y_train = np.squeeze(y_train, axis=-1)
            self.input_ = tf.placeholder('float32',shape = (None,32,32,3))
            self.input__  = tf.image.resize_images(self.input_ ,(28,28))
            self.input_x = tf.reshape(self.input__, [-1, 28, 28, 1])
            self.input_y = tf.placeholder('float32',shape = (None))
        elif dataset == "FashionMNIST":
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            self.X_train = X_train.astype(float) / 255.
            #self.Y_train = np.squeeze(y_train, axis=-1)
            self.Y_train = y_train
            # init placeholder
            self.input_ = tf.placeholder('float32',shape = (None,28,28))
            self.input_x = tf.reshape(self.input_, [-1, 28, 28, 1])
            self.input_y = tf.placeholder('float32',shape = (None))
        elif dataset == "JapaneseMNIST":
            X_train, y_train = load_japanese_mnist()
            self.X_train = X_train.astype(float) / 255.
            #self.Y_train = np.squeeze(y_train, axis=-1)
            self.Y_train = y_train
            # init placeholder
            self.input_ = tf.placeholder('float32',shape = (None,28,28))
            self.input_x = tf.reshape(self.input_, [-1, 28, 28, 1])
            self.input_y = tf.placeholder('float32',shape = (None))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # init network
        with tf.variable_scope('test_scope'):
            self.predictions = cnn_model_fn(self.input_x , mode=True)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.input_y,tf.int32),logits = self.predictions))
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.params = tf.trainable_variables(scope='test_scope')
        self.sess = tf.Session()
        self.sess.run(tf.initializers.variables(
            self.params,
            name='init'
        ))
        self.BATCH_SIZE = 32


        # init thread
        self.seed()
        #self.reset()

    def trafo(self, action):
        # trafo from state to learning_rate
        cs_0 = 0
        cs_1 = 0.5
        box_0 = self.action_space.low[0]
        box_1 = self.action_space.high[0]
        m_cs_to_box = (box_1-box_0)/(cs_1-cs_0)
        m_box_to_cs = 1/m_cs_to_box
        b_box_to_cs = cs_0 - 1/((box_1-box_0)/(cs_1-cs_0))*box_0
        cs = m_box_to_cs*action[0] + b_box_to_cs
        return [cs]

    def step(self, action):
        """ Step in gym env """
        action  = action / 5
        if not (self.hyper_space.contains(self.state + action)):
            # exited box
            loss = self.prev_loss
            reward_5 = - 0.1
        else:
            reward_5 = 0
        self.count = self.count + 1
        loss, cs_state = self.sgd_step(action)
        #reward = np.tanh(float((self.prev_loss - loss)/(self.prev_loss + 1.0e-10)))/10*(self.count/20)+add_factor/10
        reward_1 = np.tanh(float((self.prev_loss - loss)/(self.prev_loss + 1.0e-10)))/10*(self.count/10)
        reward_2 = np.linalg.norm(action)/1000*(1-self.count/10)
        reward = reward_1 + np.sign(reward_2)*max(abs(reward_2),abs(1/10*reward_1))
        done = False
        self.prev_loss = loss

        return self.obs, reward, done, {'loss': loss, 'cs_state': cs_state}

    def sgd_step(self,action):
        """ Step in GP function """
        if not (self.hyper_space.contains(self.state + action)):
            self.state = self.state
        else:
            self.state += action
        print(self.state)
        cs_state = self.trafo(self.state)
        ## state description update
        f_ = self.sgd_eval(cs_state);
        self.get_observation(f_, action)
        return f_, cs_state

    def sgd_eval(self, cs_state):
        """ Do one gradient descent step with cs state"""
        lr = cs_state[0]
        print("lr is ", str(lr))
        batch_losses = []
        batches = int(len(self.X_train) / self.BATCH_SIZE)
        for j in range(batches):
            idx = np.random.randint(self.X_train.shape[0], size=self.BATCH_SIZE)
            X_b = self.X_train[idx]
            Y_b = self.Y_train[idx]

            # train the network, note the dictionary of inputs and labels
            _, batch_loss = self.sess.run([self.train_step, self.loss], feed_dict={self.input_: X_b, self.input_y: Y_b, self.learning_rate: lr})
            batch_losses.append(batch_loss)
        batch_loss = np.mean(batch_losses)
        return batch_loss

    def get_observation(self, curr_loss, step):
        """ Transform state in invariant observation -- TO DO: invariance?? """
        #print("ddd ",str(self.prev_unscaled - curr_loss))
        test = self.prev_loss - curr_loss
        #self.obs = np.append([self.prev_loss - curr_loss, curr_loss], step)
        self.obs = np.append([self.scale(self.prev_loss - curr_loss)], self.state)
        #self.obs = np.append([self.scale(self.prev_loss - curr_loss)], step)
        # reassign the prev_loss
        return

    def scale(self,f_):
        """ Scaling -- TO DO: need for invariant scaling """
        return 10*f_ /(10+abs(f_))

    def seed(self, seed=None):
        """ Seed """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """ Reset Gym Env """

        a = self.seed()
        self.count = 0

        # init position and shape
        self.state = 0*self.hyper_space.sample()
        cs_state = self.trafo(self.state)
        self.lowest = self.sgd_eval(cs_state)
        self.first_loss = self.lowest
        self.prev_loss = self.lowest
        # init observation _ need steps to initialize
        self.step(0.1*np.ones(self.ndim))
        print(self.params)
        self.sess.run(tf.initializers.variables(
            self.params,
            name='init'
        ))
        cs_state = self.trafo(self.state)
        self.prev_loss = self.sgd_eval(cs_state)
        return self.obs
