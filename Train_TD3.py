import MyPandaReach
import gymnasium as gym
from stable_baselines3 import HerReplayBuffer,TD3
import os


env = gym.make("MyPandaReachEnv-v0")

model_dir = "models/TD3"
model_path = f"{model_dir}/500.zip"
logdir = "logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)



model = TD3(
    "MultiInputPolicy",
    env,
    batch_size=2048,
    buffer_size=1_000_000,
    gamma=0.85,
    learning_rate=0.001,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    #action_noise= OrnsteinUhlenbeckActionNoise((1,),(1,)),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    #policy_delay= 2,
    tau=0.05,
    seed=3157870761,
    verbose=1,
    tensorboard_log= logdir
)
model.learn(
    total_timesteps=10000.0,
    progress_bar=True,
    tb_log_name= "TD3"
)
model.save(f"{model_path}")
