import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import scipy.optimize as minimize

class RosenbrockEnv2(gym.Env):

    def __init__(self):
        self.a = 1.0
        self.b = 100.0
        self.min_action = np.array([-5, -5])
        self.max_action = np.array([5, 5])
        #self.min_action = -5.0
        #self.max_action = 5.0
        self.optimum_position = np.array([self.a,self.a**2]) # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        '''
        self.low_state = np.array([-1, -1, 1, 90, -30])
        self.high_state = np.array([+1, +1, 3, 100, 30])

        self.low_state = np.array([-30.0, -30.0, -30.0])
        self.high_state = np.array([+30.0, +30.0, 30.0])
        '''

        self.low_state = np.array([-1,-1,-30.0])
        self.high_state = np.array([1,1,30.0])

        #self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.num_envs = 3
        self.old = None
        self.prev_loss = 0
        self.min_loss = 10000
        self.count = 0
        self.x = 0
        self.y = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def optimum(self):
        return(np.array([self.a,self.a**2]))

    def step(self, action):
        step = action
        loss = self.rosen_grad_step(step)
        reward = (self.prev_loss - loss)/(self.prev_loss + 1.0e-5)
        reward = reward * self.count/100
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

        reward = reward - 0.01

        #print(reward)
        #time.sleep(1)

        reward = reward/2
        self.prev_loss = loss
        return self.state, reward, done, {"a": self.a, "b": self.b}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-1, high=1), self.np_random.uniform(low=-1, high=1), self.np_random.uniform(low=-30, high=30)])
        self.count = 0
        self.x = self.np_random.uniform(low=-5, high=5)
        self.y = self.np_random.uniform(low=-5, high=5)
        #rosi = self.rosen()
        self.a = self.np_random.uniform(low=0, high=3)
        self.b = self.np_random.uniform(low=4, high=30)
        #self.state[2] = 30 * rosi /(30+abs(rosi))
        self.rosen_grad_step(np.array([0,0]))
        rosi = self.rosen()/10
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        #print("position is ", str(self.x), " and ", str(self.y), "and loss is ", str(self.rosen()), "and scaled loss is ", str(30 * self.rosen() /(30+abs(self.rosen()))))
        return np.array(self.state)


    def set_state(self, state):
        print("state ", str(state))
        self.a = state[0]
        self.b = state[1]
        self.x = state[2]
        self.y = state[3]
        rosi = self.rosen()/10
        print("rosi ",str(rosi))
        self.rosen_grad_step(np.array([0,0]))
        self.prev_loss = self.min_loss = 30 * rosi /(30+abs(rosi)+1.0e-5)
        #self.state[2] = 30 * self.rosen() /(30+abs(self.rosen()))
        self.state[2] = self.prev_loss
        #self.a = self.np_random.uniform(low=1, high=3)
        #self.b = self.np_random.uniform(low=90, high=100)
        #self.state = np.array([-2.0, 2.0])
        return np.array(self.state)



    def rosen(self):
         """The Rosenbrock function"""
         #x = self.state[0:2]
         x = np.array([self.x, self.y])
         #print("IN ROSEN "+str(x))
         a = sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)
         return sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)

    def rosen_grad_step(self,beta):
        ## Calculate gradient
        dx = -2*(self.a -self.x)-4*self.b*self.x*(self.y-self.x**2)
        dy = 2*self.b*(self.y-self.x**2)
        d = np.array([dx,dy])

        #clip the gradient
        length = np.linalg.norm(d)
        dx = dx/(length+1.0e-5)
        dy = dy/(length+1.0e-5)
        '''
        if length > 30:
            dx = 30*dx/(length+1.0e-5)
            dy = 30*dy/(length+1.0e-5)
        '''



        #if np.linalg.norm(beta) < 0.01:
        #    beta = 0*beta

        # action clipping
        #
        #beta[0] = np.sgn(beta[0])*max(abs(beta[0]),0.01)
        #beta[1] = np.sgn(beta[1])*max(abs(beta[1]),0.01)


        ## step
        self.x = self.x + beta[0]/50 #*dx
        self.y = self.y + beta[1]/50 #*dy


        ## state description
        f_ = self.rosen()/10
        self.state[0] = dx;
        self.state[1] = dy;
        self.state[2] = 30 * f_ /(30+abs(f_))

        #print("position is ", str(self.x), " and ", str(self.y), "and loss is ", str(self.rosen()), "and scaled loss is ", str(30 * self.rosen() /(30+abs(self.rosen()))))
        return self.state[2]
