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

        self.low_state = np.array([1,1,-30.0])
        self.high_state = np.array([4,20,30.0])

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
        #print("action ",str(action))
        step = action
        #step = action[0]
        # get gradient step
        loss = self.rosen_grad_step(step)
        #print(loss)
        #dist = np.linalg.norm(self.state[0:2]-self.optimum())

        #done = bool(dist<0.5)
        #reward
        reward = (self.prev_loss - loss)/self.prev_loss
        reward = reward * (2-self.count/100)
        self.count = self.count + 1

        #reward = reward/100
        #reward = np.sign(reward) * min(abs(reward),100)
        #reward = np.sign(reward) * min(abs(reward),1)
        '''
        if (np.sign(reward))>0:
            reward = reward/1000
        else:
            #reward = -0.01
            reward = -0.2
        '''


        #print(self.state)
        #if done:
        #    reward = 100.0
        if abs(self.x)>30.0 or abs(self.y)>30.0:
            #
            #print("OUT")
            reward = min(2*reward,-30)
            done = True
        #elif self.count > 10:
        #    done = True
        elif abs(loss) < 10**-1 and abs(self.prev_loss) < 10**-1:
            #print("")
            #print("Whoooooppppp")
            reward = max(3* (reward + 0.1),0)
            #print(reward)
            done = False

        elif abs(loss) < 10**-1:
            reward = 3* (reward + 0.1)
            done = False
        else:
            done = False
        self.prev_loss = loss
        return self.state, reward, done, {"a": self.a, "b": self.b}

    def reset(self):
        #self.state = np.array([self.np_random.uniform(low=-1, high=1), self.np_random.uniform(low=-1, high=1),self.np_random.uniform(low=-30, high=30)])
        #self.state = np.array([0.4,0.3,2.4,96.0,3.0])
        self.a = self.np_random.uniform(low=0, high=3)
        self.b = self.np_random.uniform(low=10, high=30)
        self.state = np.array([self.a, self.b, self.np_random.uniform(low=-30, high=30)])
        self.count = 0
        self.x = self.np_random.uniform(low=-5, high=5)
        self.y = self.np_random.uniform(low=-5, high=5)
        rosi = self.rosen()
        #self.state[2] = 30 * rosi /(30+abs(rosi))
        # change assignment
        #
        #print("reset position is ", str(self.x), " and ",str(self.y), "and loss is ",str(self.rosen()))
        self.state[2] = 30 * rosi /(30+abs(rosi))
        self.prev_loss = 30 * rosi /(30+abs(rosi))
        print("position is ", str(self.x), " and ", str(self.y), "and loss is ", str(self.rosen()), "and scaled loss is ", str(30 * self.rosen() /(30+abs(self.rosen()))))
        return np.array(self.state)


    def set_state(self, state):
        self.state = state
        rosi = self.rosen()
        self.prev_loss = self.min_loss = 30 * rosi /(30+abs(rosi))
        #self.state[2] = 30 * self.rosen() /(30+abs(self.rosen()))
        self.state[2] = 30 * self.rosen() /(30+abs(self.rosen()))
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
        #d = minimize.rosen_der(self.state)
        #print(beta)
        #dx = -2*(self.a -self.state[0])-4*self.b*self.state[0]*(self.state[1]-self.state[0]**2)
        #dy = 2*self.b*(self.state[1]-self.state[0]**2)



        ##
        dx = -2*(self.a -self.x)-4*self.b*self.x*(self.y-self.x**2)
        dy = 2*self.b*(self.y-self.x**2)
        #print("")
        #print("actions ", str(beta))

        #print("gradient ", str(dx), " ", str(dy))

        d = np.array([dx,dy])

        #clip the gradient
        length = np.linalg.norm(d)
        #print("length ",str(length))
        if length > 50:
            #calculate factor
            frac = 50.0/length
            dx = frac*dx
            dy = frac*dy
        #d = np.clip(d,-10,10)
        #self.state[0:2] = self.state[0:2] - beta/10*d
        #print("position is ", str(self.x), " and ", str(self.y))
        #print("true gradient ", str(dx), " ", str(dy))
        #print("step ",str(- beta/10*dx), " ",str(- beta/10*dy))

        #print(beta)



        self.x = self.x + beta[0]/10 #*dx
        self.y = self.y + beta[1]/10 #*dy

        f_ = self.rosen()
        #self.state[2] = 30 * f_ /(30+abs(f_))
        self.state[2] = 30 * f_ /(30+abs(f_))
        #return f_
        #print("self.a is and b is  ", str(self.a), str(self.b))
        print("position is ", str(self.x), " and ", str(self.y), "and loss is ", str(self.rosen()), "and scaled loss is ", str(30 * self.rosen() /(30+abs(self.rosen()))))
        return 30 * self.rosen() /(30+abs(self.rosen()))
