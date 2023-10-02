import MyPandaReach # import env file
import gymnasium as gym
import time
from stable_baselines3 import TD3
import os
from panda_gym.pybullet import PyBullet # import customized pybullet


env = gym.make("MyPandaPathTestEnv-v0",render_mode="human")
#env  = gym.make("MyPandaReachTestEnv-v0",render_mode="human")
model_dir = "models/TD3"
logdir = "logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
model_path = f"{model_dir}/500.zip"
model = TD3.load(model_path, env=env)


episode  = 10

for episode  in range(episode):
    obs = env.reset()[0]
    done = False
    trunucated = False
    while not done and not trunucated:
        env.render()
        action, _ = model.predict(obs)
        print(action)
        obs,reward,done,trunucated,info = env.step(action)
        time.sleep(1)
env.close()