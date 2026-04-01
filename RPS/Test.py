import numpy as np
import pandas as pd
import pettingzoo
from pettingzoo.classic import tictactoe_v3
import random
import numpy as np

# A = np.array([[[1,0],[0,0],[1,0]],[[1,0],[0,1],[0,1]],[[1,0],[0,1],[0,1]]])
# print (A)
#
# A_1 = A[:,:,0]
# A_1 = A_1.transpose()
# A_2 = A[:,:,1]
# A_2 = A_2.transpose()
# print(A_1)
# print(A_2)
# print(A_1-A_2)
def obtain_state(observation, agent):
    board_state = observation['observation']
    A_1 = board_state[:, :, 0]
    A_1 = A_1.transpose()
    A_2 = board_state[:, :, 1]
    A_2 = A_2.transpose()
    if agent == "player_1":
        tag = 1
        state = A_1 - A_2
    else:
        tag = -1
        state = A_2 - A_1
    return state

def updateQtable(observation, state, Q_table):
    action_mask = observation['action_mask']
    available_actions = np.where(action_mask == 1)[0]
    if str(state) not in Q_table:
        Q_table[str(state)] = {}
        actions = available_actions
        for action in actions:
            Q_table[str(state)][str(action)] = 0
    # for state in Q_table:
    #     print(state)
    #     for action in Q_table[state]:
    #         print(action, ': ', Q_table[state][action])
    #     print('--------------')
    return Q_table

def random_action(observation):
    action_mask = observation['action_mask']
    available_actions = np.where(action_mask == 1)[0]
    random_index = random.randint(0, len(available_actions) - 1)
    return available_actions[random_index]

def main():
    Q_table = {}
    EPSILON = 0.05
    ALPHA = 0.5
    GAMMA = 1
    lastState_blue = None
    lastAction_blue = None
    lastReward_blue = None
    lastState_red = None
    lastAction_red = None
    lastReward_red = None

    env = tictactoe_v3.env(render_mode="human")
    for i in range(1):
        env.reset(seed=42)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            state = obtain_state(observation, agent)
            Q_table = updateQtable(observation, state, Q_table)
            print(Q_table)
            print(agent)
            print(reward)
            if termination or truncation:
                action = None
                print("Done")
                env.step(action)
                break
            else:
                action = random_action(observation)
            env.step(action)
        env.render()
if __name__ == "__main__":
    main()
'[[ 1 -1  0]\n [ 1  0  0]\n [-1 -1  1]]'