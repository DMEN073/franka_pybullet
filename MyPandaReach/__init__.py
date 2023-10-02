from gymnasium.envs.registration import register

register (
    id ="MyPandaReachEnv-v0",
    entry_point = "MyPandaReach.envs:PandaReachEnv",
    max_episode_steps= 50,
)

register (
    id ="MyPandaReachTestEnv-v0",
    entry_point = "MyPandaReach.envs:PandaReachTestEnv",
    max_episode_steps= 50,
)

register (
    id ="MyPandaPathTestEnv-v0",
    entry_point = "MyPandaReach.envs:PandaPathTestEnv",
    max_episode_steps= 50,
)


