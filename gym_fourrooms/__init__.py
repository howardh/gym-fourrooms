from gym.envs.registration import register

register(
    id='fourrooms-v0',
    entry_point='gym_fourrooms.envs:FourRoomsEnv',
)
