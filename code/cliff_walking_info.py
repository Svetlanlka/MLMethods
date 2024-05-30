import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from pprint import pprint

def main():
    state, action = 0, 0
    env = gym.make("CliffWalking-v0")

    print('Пространство состояний:')
    pprint(env.observation_space)
    print()
    print('Пространство действий:')
    pprint(env.action_space)
    print()
    print('Вероятности для 0 состояния:')
    pprint(env.P[state])
    print('Вероятности для 34го состояния:')
    pprint(env.P[34])
    print('Вероятности для 35го состояния:')
    pprint(env.P[35])


if __name__ == '__main__':
    main()