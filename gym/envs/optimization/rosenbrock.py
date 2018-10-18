import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import scipy.optimize as minimize

class RosenbrockEnv(gym.Env):

    def __init__(self):
        self.a = 1.0
        self.b = 100.0
        #self.min_action = np.array([-5, -5])
        #self.max_action = np.array([5, 5])
        self.min_action = 0.0
        self.max_action = 5.0
        self.optimum_position = np.array([self.a,self.a**2]) # was 0.5 in gym, 0.45 in Arnaud de Broissia's version

        self.low_state = np.array([-100, -100])
        self.high_state = np.array([+100, +100])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        #self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.old = None
        self.prev_loss = 0
        self.count = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #step = min(max(action[0], self.min_action), self.max_action)
        self.count += 1
        step = action[0]
        # get gradient step
        loss = self.rosen_grad_step(step)

        #calculate distance to optimum position
        dist = np.linalg.norm(self.state-self.optimum_position)

        done = bool(dist<0.1)
        reward = 0

        #reward
        reward = self.prev_loss - loss
        reward = reward/100
        reward = np.sign(reward) * min(abs(reward),1)-1
        self.prev_loss = loss

        if done:
            reward = 100.0
        elif abs(self.state[0])>100 or abs(self.state[1])>100 or self.count > 1000:
            reward = -100
            done = True
        '''
        elif dist<1:
            reward = 1-dist
        else:
            reward -= 1
        #print(done)
        '''

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-10, high=10), self.np_random.uniform(low=-10, high=10)])
        self.count = 0
        return np.array(self.state)


    def set_state(self, state):
        print("nsdjkdnw")
        self.state = state
        #self.state = np.array([-2.0, 2.0])
        return np.array(self.state)



    def rosen(self):
         """The Rosenbrock function"""
         x = self.state
         return sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)

    def rosen_grad_step(self,beta):
        #d = minimize.rosen_der(self.state)

        dx = -2*(self.a -self.state[0])-4*self.b*self.state[0]*(self.state[1]-self.state[0]**2)
        dy = 2*self.b*(self.state[1]-self.state[0]**2)

        d = np.array([dx,dy])

        #clip the gradient
        length = np.linalg.norm(d)
        if length > 10:
            #calculate factor
            frac = 10/length
            d = frac*d
        #d = np.clip(d,-10,10)

        self.state = self.state - beta/10*d
        f_ = self.rosen()
        return f_
