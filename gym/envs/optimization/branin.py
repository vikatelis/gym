import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import scipy.optimize as minimize

class Branin(gym.Env):

    def __init__(self):
        # Parameters of the function
        PI = 3.14159265359
        self.a = 1;
        self.b = 5.1/(4*pow(PI,2));
        self.c = 5/PI;
        self.r = 6;
        self.s = 10;
        self.t = 1/(8*PI);

        # range action
        self.min_action = np.array([-10.0, -10.0])
        self.max_action = np.array([10.0, 10.0])

        # range state
        self.low_state = np.array([-30.0,-30.0,-10.0,-10.0])
        self.high_state = np.array([30.0,30.0,10.0,10.0])

        # boxes
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        # init thread
        self.seed()
        self.reset()

    def reset(self):
        """Reset Gym Env"""

        # set initial position
        self.x = self.np_random.uniform(low=-5.0, high=10.0)
        self.y = self.np_random.uniform(low=0.0, high=15.0)

        # set initial position
        f_ = self.branin()/10

        # init state description
        # step 0
        self.branin_step(np.array([0,0]))
        f_ = self.branin()/10
        self.prev_loss = 30 * f_ /(30+abs(f_))
        self.prev_unscaled = self.unscaled

        # step 0.1
        self.branin_step(np.array([0.1,0.1]))
        f_ = self.branin()/10
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.min([np.abs(self.prev_unscaled - f_),30])
        self.prev_loss = 30 * f_ /(30+abs(f_))
        self.prev_unscaled = self.unscaled

        # step -0.1
        out = self.branin_step(np.array([-0.1,-0.1]))
        f_ = self.branin()/10
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.min([np.abs(self.prev_unscaled - f_),30])
        self.prev_loss = out
        self.prev_unscaled = self.unscaled

        #  reset counter
        self.count = 0
        return np.array(self.state)

    def seed(self, seed=None):
        """Init seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Branin Step"""
        # step size
        step = action
        # scaled loss
        loss = self.branin_step(step)
        # reward unscaled
        reward = (self.prev_unscaled - self.unscaled)/(self.prev_unscaled + 1.0e-5)
        # clip reward
        reward = np.sign(reward) * min([np.abs(reward), 10])
        # scale reward: reward scaled by absolute value and more emphasize on latter rewards!
        reward = reward * np.exp(self.count/100) * (5-min(loss,4))
        self.count = self.count + 1

        if abs(self.x)>30.0 or abs(self.y)>30.0:
            reward = min(2*reward,-10)
            done = True
        elif abs(loss) < 10**-1 and abs(self.prev_loss) < 10**-1:
            reward = max(2* (reward + 0.05),0)
            done = False
        elif abs(loss) < 10**-1:
            reward = 2* (reward + 0.05)
            done = False
        elif reward < 0:
            reward = 2* reward
            done = False
        else:
            done = False

        # scale reward by 6 - hope to decrease variance in value loss ;(
        #reward = reward/6
        self.prev_loss = loss
        self.prev_unscaled = self.unscaled
        return self.state, reward, done, {}


    def set_state(self, state):
        """Set external init state"""
        return np.array(self.state)



    def branin(self):
         """Branin Function"""
         return self.a*(self.y - self.b*self.x**2 + self.c*self.x - self.r)**2 + self.s*(1-self.t)*cos(self.x) + self.s

    def branin_step(self,beta):
        """Branin Step"""

        ##  Do step
        self.x = self.x + beta[0]/50
        self.y = self.y + beta[1]/50

        ## adapt state description
        f_ = self.branin()/10
        self.unscaled = f_
        self.state[1] = self.scaled_loss(f_)
        self.state[0] = np.sign(self.prev_unscaled - f_) * np.max([np.abs(self.prev_unscaled - f_),30])
        self.state[2] = beta[0]
        self.state[3] = beta[1]
        return self.state[1]

    def scale_loss(self,loss):
        """Scale Loss to be in certain range"""
        return 30 * loss /(30+abs(loss))
