import math
import random
from random import randint

import gym
from gym import spaces
import gym.spaces
from gym.utils import seeding
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from scipy import interpolate

class GP(gym.Env):
    def __init__(self):
        # define dimensionality
        self.ndim = 1
        # kernels
        # 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
        #                length_scale_bounds=(0.1, 10.0),
        #                periodicity_bounds=(1.0, 10.0)),
        self.kernels = [1.0 * RBF(length_scale=4.0, length_scale_bounds=(1e-1, 10.0)),
                        1.0 * RBF(length_scale=1.5, length_scale_bounds=(1e-1, 10.0)),
                        1.0 * RationalQuadratic(length_scale=2.0, alpha=2),
                        1.0 * RationalQuadratic(length_scale=1.3, alpha=2),
                        0.3 * (DotProduct(sigma_0=0, sigma_0_bounds=(0.1, 1.0))) * RationalQuadratic(length_scale=2.0, alpha=2),
                        0.3 * (DotProduct(sigma_0=0, sigma_0_bounds=(0.1, 1.0))) * RBF(length_scale=4.0, length_scale_bounds=(1e-1, 10.0)),
                        1.0 * Matern(length_scale=2.0, length_scale_bounds=(1e-1, 10.0), nu=1)]
        # range action
        self.min_action = -10*np.ones(self.ndim)
        self.max_action = 10*np.ones(self.ndim)
        # range observation - dimensionality
        self.low_state = np.append([-30.0],self.min_action)
        self.high_state = np.append([30.0],self.max_action)
        # range state - dimensionality
        self.low_state_p = -7*np.ones(self.ndim)
        self.high_state_p = 7*np.ones(self.ndim)
        # meshgrid of states
        #self.x_ = np.arange(self.low_state_p[0]-1,self.high_state_p[0]+1,1)
        self.x_ = np.arange(-8,8,0.1)
        self.grid = np.array(np.meshgrid(*[self.x_ for x in range(self.ndim)])).T.reshape(-1,self.ndim)
        # boxes
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=None, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=None, dtype=np.float32)
        self.hyper_space = spaces.Box(low=self.low_state_p, high=self.high_state_p, shape=None, dtype=np.float32)
        # init
        self.obs = self.observation_space.sample()
        self.state = self.hyper_space.sample()
        self.prev_unscaled = 0
        self.prev_loss = 0
        self.gp = 0
        self.lowest = 0
        self.scale_function = 0
        #randomly sample 4 different surfaces
        self.function_list = []
        for i in range(len(self.kernels)):
            # choose kernel randomly
            kernel = self.kernels[i]
            # Generate function with Gaussian Process
            self.gp = GaussianProcessRegressor(kernel=kernel)
            # calculate current prior and interpolate between points
            z = self.gp.sample_y(self.grid, n_samples=30, random_state=None)
            for i in range(z.shape[-1]):
                loss_func = interpolate.Rbf(*[self.grid[:,x] for x in range(self.ndim)], z[:,i])
                self.function_list.append(loss_func)
                # plot
                # if self.ndim == 1:
                #     import matplotlib.pyplot as plt
                #     plt.plot(self.grid,loss_func(self.grid))
                #     plt.show()
                # if self.ndim == 2:
                #     import matplotlib.pyplot as plt
                #     from matplotlib import cm
                #     from mpl_toolkits.mplot3d.axes3d import Axes3D
                #
                #     X,Y = np.meshgrid(*[self.x_ for x in range(self.ndim)])
                #     Z = loss_func(X, Y,)
                #     fig = plt.figure(figsize=(8, 6))
                #
                #     ax = fig.add_subplot(1, 1, 1, projection='3d')
                #
                #     ax.plot_surface(X, Y, Z, rstride=2, cstride=2)
                #     cset = ax.contour(X, Y, Z, zdir='z', offset=-4 )
                #     cset = ax.contour(X, Y, Z, zdir='x', offset=-8)
                #     cset = ax.contour(X, Y, Z, zdir='y', offset=8)
                #
                #     ax.set_xlim3d(-8, 8)
                #     ax.set_ylim3d(-8, 8)
                #     ax.set_zlim3d(-8, 8)
                #     fig.show()

        # init thread
        self.seed()
        #self.reset()

    def step(self, action):
        """ Step in gym env """
        self.count = self.count + 1
        action  = action / 10
        if not (self.hyper_space.contains(self.state+action)):
            # exited box
            loss = self.prev_loss
            reward_5 = - 0.1
        else:
            reward_5 = 0

        loss = self.gp_step(action)
        add_factor = self.lowest - loss
        #reward = np.tanh(float((self.prev_loss - loss)/(self.prev_loss + 1.0e-10)))/10*(self.count/20)+add_factor/10
        #reward_1 = np.tanh(float((self.prev_loss - loss)/max(np.abs(self.prev_loss + 1.0e-10), np.abs(loss + 1.0e-10))))
        reward_1 = np.tanh(float((self.prev_loss - loss)))
        reward_2 = np.linalg.norm(action)/1000
        reward_3 = np.tanh(add_factor)
        #reward = reward_1*(self.count/10) + np.sign(reward_2)*max(abs(reward_2),abs(1/10*reward_1))*(1-self.count/10) + reward_3
        reward = reward_1*(self.count/20) + reward_3*(self.count/20) + reward_5

        done = False
        if loss < self.lowest:
            self.lowest = loss


        self.prev_loss = loss
        self.prev_unscaled = loss

        return self.obs, reward, done, {'loss': loss}

    def gp_step(self,action):
        """ Step in GP function """
        if (self.hyper_space.contains(self.state+action)):
            self.state += action
        ## state description update
        f_ = self.gp_eval();
        self.get_observation(f_, action)
        return f_

    def gp_eval(self):
        """ Evaluate function generated by self.gp """
        # get covariance of prev state and current state
        #y_samples = self.gp.sample_y(test)
        y_samples = self.scale_function*self.loss_func(*self.state)
        #y_samples = self.gp.sample_y(test,1, None)
        return y_samples

    def get_observation(self, curr_loss, step):
        """ Transform state in invariant observation -- TO DO: invariance?? """
        #print("ddd ",str(self.prev_unscaled - curr_loss))
        test = self.prev_loss - curr_loss
        #self.obs = np.append([self.prev_loss - curr_loss, curr_loss], step)
        self.obs = np.append([self.scale(self.prev_loss - curr_loss)], self.state)
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
        self.gp = 0
        self.scale_function = random.uniform(0, 5)
        #self.loss_func = 0
        self.count = 0
        # randomly sample from a
        no = randint(0, len(self.function_list))
        self.loss_func = self.function_list[no]
        # init position and shape
        self.state = 0*self.hyper_space.sample()
        self.lowest = self.gp_eval()
        self.first_loss = self.lowest
        self.prev_loss = self.lowest
        # init observation _ need steps to initialize
        self.step(0.1*np.ones(self.ndim))
        self.prev_loss = self.gp_eval()
        if self.prev_loss < self.lowest:
            self.lowest = self.prev_loss
        # TODO: RETURN self ons and not self.state ??
        #  reset counter
        self.count = 0
        return self.obs
        #return np.array(self.state)
