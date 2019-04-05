import math
import random
from random import randint

import gym
from gym import spaces
import gym.spaces
from gym.utils import seeding
from gym.utils.cnn_standard import cnn_model_fn
from gym.utils.load_data import sample_dataset
import numpy as np
import tensorflow as tf
from gym.utils import CNNPrototypes

class SGDwithSampledCNN(gym.Env):
    def __init__(self):
        # define dimensionality
        self.ndim = 1
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
        self.num_batches = 1900
        self.batch_window = 130
        self.BATCH_SIZE = 32
        self.CNNPrototypes = CNNPrototypes.CNNPrototypes()
        # init thread
        self.seed()

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
        done = False
        loss = self.sgd_step(action)
        add_factor = self.lowest - loss
        reward_1 = np.tanh(float((self.prev_loss - loss)))
        reward_2 = np.tanh(np.linalg.norm(action))
        reward_3 = np.tanh(add_factor)
        reward = reward_1 + reward_3 + reward_5
        self.prev_loss = loss

        return self.obs, reward, done, {'loss': loss}

    def sgd_step(self,action):
        """ Step in GP function """
        if not (self.hyper_space.contains(self.state + action)):
            self.state = self.state
        else:
            self.state += action
        cs_state = self.trafo(self.state)
        ## state description update
        f_ = self.sgd_eval(cs_state);
        self.get_observation(f_, action)
        return f_

    def sgd_eval(self, cs_state):
        """ Do one gradient descent step with cs state"""
        lr = cs_state[0]
        print("lr is ", str(lr))
        batch_losses = []
        for j in range(self.batch_window):
            print("Batch no ",str(j))
            idx = np.random.randint(self.X_train.shape[0], size=self.BATCH_SIZE)
            X_b = self.X_train[idx]
            Y_b = self.Y_train[idx]
            # train the network, note the dictionary of inputs and labels
            _, batch_loss = self.sess.run([self.train_step, self.loss], feed_dict={self.input_: X_b, self.input_y: Y_b, self.learning_rate: lr})
            batch_losses.append(batch_loss)
        batch_loss = np.mean(batch_losses)
        print(batch_loss)
        return batch_loss

    def forward_pass(self):
        """ Evaluate current loss"""
        idx = np.random.randint(self.X_train.shape[0], size=self.batch_window*self.BATCH_SIZE)
        X_b = self.X_train[idx]
        Y_b = self.Y_train[idx]
        # sample subset of size Batch_Size*batch_window
        total_loss = self.sess.run([self.loss], feed_dict={self.input_: X_b , self.input_y: Y_b})
        return total_loss

    def get_observation(self, curr_loss, step):
        """ Transform state in invariant observation -- TO DO: invariance?? """
        self.obs = np.append([self.scale(self.prev_loss - curr_loss)], step)
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
        print("reset")
        # tf.keras.backend.clear_session()
        a = self.seed()
        # sample DataSet
        self.X_train, self.Y_train, type = sample_dataset("CIFAR10")
        self.BATCH_SIZE = int(len(self.X_train)/self.num_batches)
        print("Dataset ", str(type), "size", str(len(self.X_train)))
        # preprocess DataSet
        if type == "MNIST" or type == "JapaneseMNIST" or type == "FashionMNIST":
            self.input_ = tf.placeholder('float32',shape = (None,28,28))
            self.input_x = tf.reshape(self.input_, [-1, 28, 28, 1])
            self.input_y = tf.placeholder('float32',shape = (None))
        elif type == "CIFAR10" or type == "SVHN":
            self.input_ = tf.placeholder('float32',shape = (None,32,32,3))
            self.input_resize  = tf.image.resize_images(self.input_ ,(28,28))
            self.input_x = tf.reshape(self.input_resize, [-1, 28, 28, 3])
            self.input_y = tf.placeholder('float32',shape = (None))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.predictions, tag, self.input_x, self.input_y, self.loss, self.train_step, self.learning_rate, self.input_  = \
            self.CNNPrototypes.get_network(self.input_x, self.input_, self.input_y, self.learning_rate, id=10, nr_classes=10, mode=True)
        #tf.get_default_graph().get_operations()

        # init network
        self.params = tf.trainable_variables(scope="cnn_scope_"+tag)

        self.sess = tf.Session()
        self.sess.run(tf.initializers.variables(
            self.params,
            name='init'
        ))

        # init first loss
        print("doing forward step")
        self.lowest = self.forward_pass()
        print("loss is what ", str(self.lowest))
        self.prev_loss = self.lowest
        # init state
        self.state = 0*self.hyper_space.sample()
        self.obs = 0*self.observation_space.sample()
        print("reset done")
        return self.obs
