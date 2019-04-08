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
import multiprocessing
import math

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
        self.batch_window = 12
        self.BATCH_SIZE = 32
        process = multiprocessing.current_process()
        self.thread_id = process.pid
        self.CNNPrototypes = CNNPrototypes.CNNPrototypes(self.thread_id)
        tf_graph = tf.Graph()
        self.tf_session = tf.Session(graph=tf_graph)
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
            reward_5 = - 0.01
        else:
            reward_5 = 0
        done = False
        loss, done = self.sgd_step(action)
        add_factor = self.lowest - loss
        reward_1 = np.tanh(float((self.prev_loss - loss)))
        reward_2 = np.tanh(np.linalg.norm(action))
        reward_3 = np.tanh(add_factor)
        if done == True:
            reward = -10
        else:
            reward = reward_1 + reward_3 + reward_5
        self.prev_loss = loss

        if loss < self.lowest:
            self.lowest = loss

        return self.obs, reward, done, {'loss': loss}

    def sgd_step(self,action):
        """ Step in GP function """
        if not (self.hyper_space.contains(self.state + action)):
            self.state = self.state
        else:
            self.state += action
        cs_state = self.trafo(self.state)
        ## state description update
        f_, done = self.sgd_eval(cs_state);
        self.get_observation(f_, action)
        return f_, done

    def sgd_eval(self, cs_state):
        """ Do one gradient descent step with cs state"""
        lr = cs_state[0]
        batch_losses = []
        done = False
        for j in range(self.batch_window):
            idx = np.random.randint(self.X_train.shape[0], size=self.BATCH_SIZE)
            X_b = self.X_train[idx]
            Y_b = self.Y_train[idx]
            # train the network, note the dictionary of inputs and labels
            #with self.tf_session.as_default(), self.tf_session.graph.as_default() as sess:
            _, batch_loss = self.tf_session.run([self.train_step, self.loss], feed_dict={self.input_: X_b, self.input_y: Y_b, self.learning_rate: lr})
            if batch_loss > 100 or math.isnan(batch_loss):
                print(batch_loss)
                batch_loss = 100
                done = True
                break
            batch_losses.append(batch_loss)

        if done == True:
            batchloss = 100
        else:
            batch_loss = np.mean(batch_losses)

        return batch_loss, done

    def forward_pass(self):
        """ Evaluate current loss"""
        idx = np.random.randint(self.X_train.shape[0], size=self.batch_window*self.BATCH_SIZE)
        X_b = self.X_train[idx]
        Y_b = self.Y_train[idx]
        # sample subset of size Batch_Size*batch_window
        #with self.tf_session.as_default(), self.tf_session.graph.as_default() as sess:

        total_loss = self.tf_session.run([self.loss], feed_dict={self.input_: X_b , self.input_y: Y_b})
        return total_loss[0]

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
        #tf.keras.backend.clear_session()
        self.tf_session.close()
        a = self.seed()
        # sample DataSet
        self.X_train, self.Y_train, type, nr_classes = sample_dataset()
        self.BATCH_SIZE = int(len(self.X_train)/self.num_batches)
        # preprocess DataSet
        tf_graph = tf.Graph()
        self.tf_session = tf.Session(graph=tf_graph)
        with tf_graph.as_default(), self.tf_session.as_default(), self.tf_session.graph.as_default():

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

            model_id = np.random.randint(1, 10)
            self.predictions = self.CNNPrototypes.get_network(self.input_x, self.input_, self.input_y, self.learning_rate, id=model_id, nr_classes=nr_classes, mode=True)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.input_y,tf.int32),logits = self.predictions))
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            init_graph = tf.global_variables_initializer()

        self.tf_session.run(init_graph)

        # init first loss
        self.lowest = self.forward_pass()
        self.prev_loss = self.lowest
        # init state
        self.state = 0*self.hyper_space.sample()
        self.obs = 0*self.observation_space.sample()
        return self.obs
