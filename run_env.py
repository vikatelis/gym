#import logistic_regression
import tensorflow as tf
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
print("heloo are you alive?")
import numpy as np
import gym


# init training env
print("init env")
#env_type, env_id = get_env_type("LogReg-v0")
env_type, env_id = "optimization", "SGDwithSampledCNN-v0"
env = gym.make(env_id)
print(env)
env.num_envs = 1
#ßenv = make_rosenbrock_env(env_id, 1, None)
obs = env.reset()
ob_space = env.observation_space
print(ob_space.shape)
ac_space = env.action_space
print(ac_space.shape)


step = 0
done = False
obs = env.reset()
while done == False and step<=20:
    actions = 0.1
    obs, _, done, info  = env.step(actions)
    print("loss: ", str(info['loss']))
    #print("config: ", str(info['config']))
    done = done.any() if isinstance(done, np.ndarray) else done

    if done:
        print(done)
        obs = env.reset()
'''
# run it
out = logReg.objective_function([-3,1,20,0.0], fold=10)
print(type(out))
print(out['function_value'])
'''
