import pytest
import gym_fourrooms
from gym_fourrooms.envs import FourRoomsEnv

def test_init_reset_step():
    env = FourRoomsEnv()
    env.reset()
    env.step(env.action_space.sample())

def test_goal_change_every_step():
    env = FourRoomsEnv(goal_duration_steps=1)
    env.reset()
    g0 = env.goal
    env.step(env.action_space.sample())
    g1 = env.goal
    env.step(env.action_space.sample())
    g2 = env.goal
    assert g0 is not g1
    assert g1 is not g2

def test_goal_change_every_2_step():
    env = FourRoomsEnv(goal_duration_steps=2)
    env.reset()
    g0 = env.goal
    env.step(env.action_space.sample())
    g1 = env.goal
    env.step(env.action_space.sample()) # Goal change here
    g2 = env.goal
    env.reset()
    g3 = env.goal
    env.step(env.action_space.sample())
    g4 = env.goal
    env.step(env.action_space.sample()) # Goal change here
    g5 = env.goal
    assert g0 is g1
    assert g1 is not g2
    assert g2 is g3
    assert g3 is g4
    assert g4 is not g5

def test_goal_change_every_episode():
    env = FourRoomsEnv(goal_duration_episodes=1)
    env.reset()
    g0 = env.goal
    env.step(env.action_space.sample())
    g1 = env.goal
    env.step(env.action_space.sample())
    g2 = env.goal
    env.reset() # Goal change here
    g3 = env.goal
    env.step(env.action_space.sample())
    g4 = env.goal
    assert g0 is g1
    assert g1 is g2
    assert g2 is not g3
    assert g3 is g4

def test_goal_never_change():
    env = FourRoomsEnv(goal_duration_episodes=float('inf'))
    env.reset()
    g0 = env.goal
    env.step(env.action_space.sample())
    g1 = env.goal
    env.step(env.action_space.sample())
    g2 = env.goal
    env.reset()
    g3 = env.goal
    env.step(env.action_space.sample())
    g4 = env.goal
    assert g0 is g1
    assert g1 is g2
    assert g2 is g3
    assert g3 is g4

def test_seed():
    env0 = FourRoomsEnv()
    env0.seed(0)
    env1 = FourRoomsEnv()
    env1.seed(0)

    term = True

    for _ in range(10):
        if term:
            x0 = env0.reset()
            x1 = env1.reset()
            assert (x0 == x1).all()
        else:
            a0 = env0.action_space.sample()
            a1 = env1.action_space.sample()
            assert a0 == a1

            x0,r,term,_ = env0.step(a0)
            x1,r,term,_ = env1.step(a1)
            assert (x0 == x1).all()

def test_state_dict():
    env0 = FourRoomsEnv()
    env1 = FourRoomsEnv()
    env2 = FourRoomsEnv()

    state = env0.state_dict()

    def sample_chain(env, steps):
        output = []
        term = True
        for _ in range(steps):
            if term:
                x = env.reset()
                output.append(x)
            else:
                a = env.action_space.sample()
                output.append(a)
                x,r,term,_ = env.step(a)
                output.append(x)
        return output

    x0 = sample_chain(env0,10)

    env1.load_state_dict(state)
    x1 = sample_chain(env1,10)
    assert all([(a == b).all() for a,b in zip(x0,x1)])

    env1.load_state_dict(state)
    x1 = sample_chain(env1,10)
    assert all([(a == b).all() for a,b in zip(x0,x1)])

    x2 = sample_chain(env2,10)
    assert any([(a != b).any() for a,b in zip(x0,x2)])

    env2.load_state_dict(state)
    x2 = sample_chain(env2,10)
    assert all([(a == b).all() for a,b in zip(x0,x2)])
