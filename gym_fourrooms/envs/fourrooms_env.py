import gym
from gym.utils import seeding
import numpy as np

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
def string_to_bool_map(str_map):
    bool_map = []
    for row in str_map.split('\n')[1:]:
        bool_map.append([r==' ' for r in row])
    return np.array(bool_map)
env_map = string_to_bool_map(four_rooms_map)

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
    def __init__(self,fail_prob=1/3, env_map=env_map,
            goal_duration_steps=None, goal_duration_episodes=None):
        # Process env map
        if type(env_map) is str:
            self.env_map = string_to_bool_map(env_map)
        else:
            self.env_map = env_map
        self.coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if self.env_map[y,x]:
                    self.coords.append(np.array([y,x]))

        # Process other params
        self.fail_prob = fail_prob
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.array([0,0]),high=np.array(self.env_map.shape))

        if goal_duration_steps is None and goal_duration_episodes is None:
            goal_duration_episodes = 1
        elif goal_duration_steps is not None and goal_duration_episodes is not None:
            raise ValueError('Both goal_duration_steps and goal_duration_episodes were assigned values. Only one can be used at a time.')
        self.goal_duration_steps = goal_duration_steps
        self.goal_duration_episodes = goal_duration_episodes
        self.step_count = 0
        self.episode_count = 0

        self.pos = None
        self.goal = None

        self.seed()

    def step(self, action):
        # Update state
        if self.rand.rand() < self.fail_prob*4/3:
            action = self.action_space.sample()
        p = self.pos+directions[action]
        if self.env_map[p[0],p[1]]:
            self.pos = p
        obs = np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        if (self.pos == self.goal).all():
            reward = 1
            done = True
            self.pos = None
        else:
            reward = 0
            done = False
        info = {}
        # Update counts
        if self.goal_duration_steps is not None:
            self.step_count += 1
            if self.step_count >= self.goal_duration_steps:
                self.step_count = 0
                self.reset_goal()
        # Return updated states
        return obs, reward, done, info

    def reset(self):
        self.reset_pos()
        # Check of goal state needs to be changes
        if self.goal is None:
            self.reset_goal()
        if self.goal_duration_steps is None and self.goal_duration_episodes is None:
            self.reset_goal()
        elif self.goal_duration_episodes is not None:
            if self.episode_count >= self.goal_duration_episodes:
                self.reset_goal()
                self.episode_count = 0
            self.episode_count += 1
        return np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])

    def reset_goal(self, goal=None):
        if goal is not None:
            self.goal = goal[:]
        else:
            if self.pos is None:
                goal_index = self.rand.randint(0,len(self.coords))
            else:
                goal_index = self.rand.randint(0,len(self.coords)-1)
                if (self.coords[goal_index] == self.pos).all():
                    goal_index = len(self.coords)-1
            self.goal = self.coords[goal_index][:]

    def reset_pos(self):
        if self.goal is None:
            pos_index = self.rand.randint(0,len(self.coords))
        else:
            pos_index = self.rand.randint(0,len(self.coords)-1)
            if (self.coords[pos_index] == self.goal).all():
                pos_index = len(self.coords)-1
        self.pos = self.coords[pos_index][:]

    def seed(self, seed=None):
        self.rand = np.random.RandomState(seed)
        self.action_space.np_random = self.rand

    def render(self, mode='human'):
        print_state(self.env_map,self.pos,self.goal)

    def close(self):
        self.pos = None
        self.goal = None

    def state_dict(self):
        return {
                'env_map': self.env_map,
                'coords': self.coords,
                'fail_prob': self.fail_prob,
                'action_space': self.action_space,
                'observation_space': self.observation_space,
                'goal_duration_episodes': self.goal_duration_episodes,
                'goal_duration_steps': self.goal_duration_steps,
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'pos': self.pos,
                'goal': self.goal,
                'rand': self.rand.get_state()
        }

    def load_state_dict(self,state):
        self.env_map = state['env_map']
        self.coords = state['coords']
        self.fail_prob = state['fail_prob']
        self.action_space = state['action_space']
        self.observation_space = state['observation_space']
        self.goal_duration_steps = state['goal_duration_steps']
        self.goal_duration_episodes = state['goal_duration_episodes']
        self.episode_count = state['episode_count']
        self.step_count = state['step_count']
        self.pos = state['pos']
        self.goal = state['goal']
        self.rand.set_state(state['rand'])
        self.action_space.np_random = self.rand
