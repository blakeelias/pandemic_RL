from gym.envs.registration import register

register(
    id='pandemic-v0',
    entry_point='gym_pandemic.envs:PandemicEnv',
)
