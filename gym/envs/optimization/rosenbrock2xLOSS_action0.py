import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import scipy.optimize as minimize

class RosenbrockEnv2(gym.Env):

    def __init__(self):
        """Inite Env"""
        # init param
        self.a = 1.0
        self.b = 100.0

        # action space
        self.min_action = np.array([-10.0, -10.0])
        self.max_action = np.array([10.0, 10.0])

        # state space
        self.low_state = np.array([-30.0,-30.0,-10.0,-10.0])
        self.high_state = np.array([30.0,30.0,10.0,10.0])

        # boxes
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        # init prevs
        self.prev_loss = 0
        self.unscaled = 0
        self.prev_unscaled = 0
        self.min_loss = 10000
        self.count = 0
        self.x = 0
        self.y = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        """Seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Rosi step"""
        # step size
        step = action
        # scaled loss
        loss = self.rosen_grad_step(step)
        # reward unscaled
        reward = (self.prev_unscaled - self.unscaled)/(self.prev_unscaled + 1.0e-5)
        # clip reward
        reward = np.sign(reward) * min([np.abs(reward), 10])
        # scale reward: reward scaled by absolute value and more emphasize on latter rewards!
        reward = reward * np.exp(self.count/100) * (5-min(loss,4))
        self.count = self.count + 1
        done = False

        if abs(self.x)>30.0 or abs(self.y)>30.0:
            reward = min(2*reward,-10)
            done = True
        elif abs(loss) < 10**-1 and abs(self.prev_loss) < 10**-1:
            reward = max(2* (reward + 0.05),0)
            done = False
        elif abs(loss) < 1: #10**-1:
            reward = 2* (reward + 0.05)
            done = False
        elif reward < 0:
            reward = 2* reward
            done = False
        else:
            done = False

        # scale reward by 6 - hope to decrease variance in value loss ;(
        reward = reward/4
        self.prev_loss = loss
        self.prev_unscaled = self.unscaled
        return self.state, reward, done, {'x': self.x, 'y': self.y, 'z': 10*self.unscaled}

    def reset(self):
        """Reset Gym Env"""
        self.count = 0

        # init position and shape
        self.state = np.array([self.np_random.uniform(low=-30, high=30),self.np_random.uniform(low=-30, high=30),self.np_random.uniform(low=-5, high=5),self.np_random.uniform(low=-5, high=5)])
        self.x = self.np_random.uniform(low=-6, high=6)
        self.y = self.np_random.uniform(low=-6, high=6)
        self.a = self.np_random.uniform(low=0, high=3)
        self.b = self.np_random.uniform(low=4, high=50)

        # init state description
        # step 0
        self.rosen_grad_step(np.array([0,0]))
        rosi = self.rosen() /10
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        self.prev_unscaled = self.unscaled

        # step 0.1
        rosi = self.rosen_grad_step(np.array([0.1,0.1]))
        f_ = self.rosen() /10
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.min([np.abs(self.prev_unscaled - f_),30])
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        self.prev_unscaled = self.unscaled

        # step - 0.1
        out = self.rosen_grad_step(np.array([-0.1,-0.1]))
        f_ = self.rosen() /10
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.min([np.abs(self.prev_unscaled - f_),30])
        self.prev_loss = out
        self.prev_unscaled = self.unscaled

        #  reset counter
        self.count = 0
        return np.array(self.state)


    def set_state(self, state):
        """Set External Init"""
        print("state ", str(state))
        self.a = state[0]
        self.b = state[1]
        self.x = state[2]
        self.y = state[3]
        rosi = self.rosen() /10
        self.rosen_grad_step(np.array([0,0]))
        rosi = self.rosen() /10
        self.state[0] = self.prev_loss - self.state[1]
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        self.prev_unscaled = self.unscaled
        rosi = self.rosen_grad_step(np.array([1,1]))
        self.state[0] = self.prev_loss - self.state[1]
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        self.prev_unscaled = self.unscaled
        out = self.rosen_grad_step(np.array([-1,-1]))
        self.state[0] = self.prev_loss - self.state[1]
        self.prev_loss = out
        self.prev_unscaled = self.unscaled
        return np.array(self.state)



    def rosen(self):
         """The Rosenbrock function"""
         x = np.array([self.x, self.y])
         a = sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)
         return sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)

    def rosen_grad_step(self,beta):
        """Step in Environment"""

        ## Calculate gradient
        dx = -2*(self.a -self.x)-4*self.b*self.x*(self.y-self.x**2)
        dy = 2*self.b*(self.y-self.x**2)

        self.x = self.x + beta[0]/50;
        self.y = self.y + beta[1]/50;

        ## state description update
        f_ = self.rosen() /10;
        self.unscaled = f_;
        self.state[1] = 30 * f_ /(30+abs(f_));
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.min([np.abs(self.prev_unscaled - f_),30])
        self.state[2] = beta[0]
        self.state[3] = beta[1]
        return self.state[1]
