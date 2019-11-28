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
