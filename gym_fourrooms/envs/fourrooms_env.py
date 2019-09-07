import gym
from gym.utils import seeding
import numpy as np

# TODO: Seed RNG

four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""
env_map = []
for row in four_rooms_map.split('\n')[1:]:
    env_map.append([r==' ' for r in row])
env_map = np.array(env_map)

directions = [
        np.array([-1,0]),
        np.array([0,1]),
        np.array([1,0]),
        np.array([0,-1])
]

def print_state(m,pos=None,goal=None):
    size = m.shape
    for y in range(size[0]):
        for x in range(size[1]):
            if (pos == (y,x)).all():
                print('A ',end='')
            elif (goal == (y,x)).all():
                print('G ',end='')
            elif m[y,x]:
                print('  ',end='')
            else:
                print('X ',end='')
        print()

class FourRoomsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,fail_prob=1/3, env_map=env_map):
        self.env_map = env_map
        self.coords = []
        for y in range(env_map.shape[0]):
            for x in range(env_map.shape[1]):
                if env_map[y,x]:
                    self.coords.append(np.array([y,x]))

        self.fail_prob = fail_prob
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.array([0,0]),high=np.array(env_map.shape))

    def step(self, action):
        if np.random.rand() < self.fail_prob*4/3:
            action = self.action_space.sample()
        p = self.pos+directions[action]
        if self.env_map[p[0],p[1]]:
            self.pos = p
        obs = np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        if (self.pos == self.goal).all():
            reward = 1
            done = True
            self.pos = None
            self.goal = None
        else:
            reward = 0
            done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        pos_index = np.random.randint(0,len(self.coords))
        goal_index = np.random.randint(0,len(self.coords))
        if goal_index >= pos_index:
            goal_index += 1
        self.pos = self.coords[pos_index][:]
        self.goal = self.coords[goal_index][:]
        return np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])

    def render(self, mode='human'):
        print_state(self.env_map,self.pos,self.goal)

    def close(self):
        self.pos = None
        self.goal = None
