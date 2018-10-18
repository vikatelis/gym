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
        #self.min_action = np.array([-5, -5])
        #self.max_action = np.array([5, 5])
        self.min_action = -5.0
        self.max_action = 5.0
        self.optimum_position = np.array([self.a,self.a**2]) # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        '''
        self.low_state = np.array([-1, -1, 1, 90, -20])
        self.high_state = np.array([+1, +1, 3, 100, 20])
        '''
        self.low_state = np.array([-30.0, -30.0, -20.0])
        self.high_state = np.array([+30.0, +30.0, 20.0])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        #self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.num_envs = 4
        self.old = None
        self.prev_loss = 0
        self.min_loss = 10000
        self.count = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def optimum(self):
        return(np.array([self.a,self.a**2]))

    def step(self, action):
        #step = min(max(action[0], self.min_action), self.max_action)
        #print(action)
        self.count += 1
        step = action[0]
        # get gradient step
        loss = self.rosen_grad_step(step)
        #print(loss)
        dist = np.linalg.norm(self.state[0:2]-self.optimum())

        #done = bool(dist<0.5)
        reward = 0

        #reward
        reward = self.prev_loss - loss
        reward = reward/1000
        #reward = np.sign(reward) * min(abs(reward),100)
        reward = np.sign(reward) * min(abs(reward),0.5)
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
        if abs(self.state[0])>30.0 or abs(self.state[1])>30.0:
            #
            print("OUT")
            reward = -5
            done = True
        #elif self.count > 10:
        #    done = True
        elif abs(loss) < 10**-1 and abs(self.prev_loss) < 10**-1:
            print("")
            print("Whoooooppppp")
            reward = 2 + reward
            done = False
        elif abs(loss) < 10**-1:

            print("   ")
            print("Made it ")
            reward = 1 + reward
            '''
            if reward > 0.099 and reward>0:
                reward = 5*reward
            elif reward > 0:
                reward = 100*reward
            print(reward)
            '''
            done = False

        elif reward <= 0 and abs(loss) > 10 :
            reward = min(reward,-0.5)
            done = False
        elif loss < 10:
            #reward += 0.5
            #print("In loss < 10")
            reward = 1.5*reward
            done = False
        else:
            done = False
        '''
        elif loss < 10 and loss < self.min_loss:
            self.min_loss = loss
            reward = 10
            done = False
        '''

        '''
        elif dist<1:
            reward = 1-dist
        else:
            reward -= 1
        #print(done)
        '''
        #print(reward)
        self.prev_loss = loss
        #print(reward)
        return self.state, reward, done, {}

    def reset(self):
        #
        #print("reset")
        self.state = np.array([self.np_random.uniform(low=-1, high=1), self.np_random.uniform(low=-1, high=1),self.np_random.uniform(low=-20, high=20)])
        #self.state = np.array([0.4,0.3,2.4,96.0,3.0])
        self.count = 0
        rosi = self.rosen()
        #print(abs(rosi))
        #print(10+abs(rosi))
        #print(20*rosi /(20+abs(rosi)))
        self.state[2] = 20 * rosi /(20+abs(rosi))
        self.a = self.np_random.uniform(low=1, high=3)
        self.b = self.np_random.uniform(low=90, high=100)
        self.prev_loss = 20 * rosi /(20+abs(rosi))
        return np.array(self.state)


    def set_state(self, state):
        self.state = state
        rosi = self.rosen()
        self.prev_loss = self.min_loss = 20 * rosi /(20+abs(rosi))
        self.state[2] = 20 * self.rosen() /(20+abs(self.rosen()))
        self.a = self.np_random.uniform(low=1, high=3)
        self.b = self.np_random.uniform(low=90, high=100)
        #self.state = np.array([-2.0, 2.0])
        return np.array(self.state)



    def rosen(self):
         """The Rosenbrock function"""
         x = self.state[0:2]
         #print("IN ROSEN "+str(x))

         a = sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)
         return sum(self.b*(x[-1:]-x[:1]**2.0)**2.0 + (self.a-x[:1])**2.0)

    def rosen_grad_step(self,beta):
        #d = minimize.rosen_der(self.state)
        #print(beta)
        dx = -2*(self.a -self.state[0])-4*self.b*self.state[0]*(self.state[1]-self.state[0]**2)
        dy = 2*self.b*(self.state[1]-self.state[0]**2)

        d = np.array([dx,dy])

        #clip the gradient
        length = np.linalg.norm(d)
        if length > 50:
            #calculate factor
            frac = 50/length
            d = frac*d
        #d = np.clip(d,-10,10)
        self.state[0:2] = self.state[0:2] - beta/10*d
        f_ = self.rosen()
        #time.sleep(1)
        #print(f_)
        self.state[2] = 20 * f_ /(20+abs(f_))
        #return f_
        return 20 * self.rosen() /(20+abs(self.rosen()))
