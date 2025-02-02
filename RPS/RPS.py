import pettingzoo
from pettingzoo.classic import tictactoe_v3
import random
import numpy as np
import pickle

def obtain_state(observation, agent):
    board_state = observation['observation']
    A_1 = board_state[:, :, 0]
    A_1 = A_1.transpose()
    A_1 = np.array(A_1)
    A_2 = board_state[:, :, 1]
    A_2 = A_2.transpose()
    A_2 = np.array(A_2)
    # if agent == "player_1":
    #     state = A_1
    # else:
    #     state = A_2
    if agent == "player_1":
        tag = 1
        state = A_1 - A_2
    else:
        tag = -1
        state = A_1 - A_2
    return state

def random_action(observation):
    action_mask = observation['action_mask']
    available_actions = np.where(action_mask == 1)[0]
    random_index = random.randint(0, len(available_actions) - 1)
    return available_actions[random_index]

def addNewstate(observation, state, Q_table):
    action_mask = observation['action_mask']
    available_actions = np.where(action_mask == 1)[0]
    if str(state) not in Q_table:  # Add new state to Q_table
        Q_table[str(state)] = {}
        actions = available_actions
        for action in actions:
            Q_table[str(state)][str(action)] = 0
    return Q_table
        # for state in Q_table:
        #     print(state)
        #     for action in Q_table[state]:
        #         print(action, ': ', Q_table[state][action])
        #     print('--------------')

def epsilon_greedy(Q_table, state, EPSILON, agent):
    Q_Sa = Q_table[str(state)]
    maxAction, maxValue, otherAction = [], -100, []
    for one in Q_Sa:
        if Q_Sa[one] > maxValue:
            maxValue = Q_Sa[one]
    for one in Q_Sa:
        if Q_Sa[one] == maxValue:
            maxAction.append(one)
        else:
            otherAction.append(one)
    try:
        action_pos = random.choice(maxAction) if random.random() > EPSILON else random.choice(otherAction)
    except:
        action_pos = random.choice(maxAction)
    action = {'mark': agent, 'pos': action_pos}
    return action

def updateQtable(agent,termination,truncation,observation,env,state,lastState_blue,lastState_red,lastAction_blue,lastAction_red,lastReward_blue,lastReward_red,Q_table,ALPHA,GAMMA):

    judge = (agent == 'player_1' and lastState_blue is None) or \
            (agent == 'player_2' and lastState_red is None)
    if judge:
        Q_table = addNewstate(observation, state, Q_table)
        return Q_table

    # if termination or truncation:
    #     for one in ['player_1','player_2']:
    #         S = lastState_blue if one == 'player_1' else lastState_red
    #         a = lastAction_blue if one == 'player_1' else lastAction_red
    #         R = lastReward_blue if one == 'player_1' else lastReward_red
    #         print("termination or truncation:")
    #         print('lastState S:\n', S)
    #         print('lastAction a: ', a)
    #         print('lastReward R: ', R)
    #         maxQ_S_a = 0
    #         Q_table[str(S)][str(a)] = (1 - ALPHA) * Q_table[str(S)][str(a)] \
    #                                        + ALPHA * (R + GAMMA * maxQ_S_a)
    #         print('Q(S,a) = ', Q_table[str(S)][str(a)])
    #     return Q_table

    Q_table = addNewstate(observation, state, Q_table)
    print("Normal Process:")
    S_ = state
    S = lastState_blue if agent == 'player_1' else lastState_red
    a = lastAction_blue if agent == 'player_1' else lastAction_red
    R = lastReward_blue if agent == 'player_1' else lastReward_red
    Q_S_a = Q_table[str(S_)]
    maxQ_S_a = -10
    for one in Q_S_a:
        if Q_S_a[one] > maxQ_S_a:
            maxQ_S_a = Q_S_a[one]
    print('lastState S:\n', S)
    print('State S_:\n', S_)
    print('lastAction a: ', a)
    print('lastReward R: ', R)
    Q_table[str(S)][str(a)] = (1 - ALPHA) * Q_table[str(S)][str(a)] \
                                   + ALPHA * (R + GAMMA * maxQ_S_a)
    print('Q(S,a) = ', Q_table[str(S)][str(a)])
    print('\n')

    # for state in Q_table:
        # print(state)
        # for action in Q_table[state]:
        #     print(action, ': ', Q_table[state][action])
        # print('--------------')
    return Q_table

def reset_state(lastState_blue,lastAction_blue,lastReward_blue,lastState_red,lastAction_red,lastReward_red):
    lastState_blue = None
    lastAction_blue = None
    lastReward_blue = None
    lastState_red = None
    lastAction_red = None
    lastReward_red = None
    return lastState_blue, lastAction_blue, lastReward_blue, lastState_red, lastAction_red, lastReward_red


def main():
    Q_table = {}
    tag_Q = []
    tag = 0
    EPSILON = 0.05
    ALPHA = 0.2
    GAMMA = 1
    lastState_blue = None
    lastAction_blue = None
    lastReward_blue = None
    lastState_red = None
    lastAction_red = None
    lastReward_red = None

    env = tictactoe_v3.env(render_mode="human")
    env.reset(seed=42)
    for i in range(10):
        print(f"times: {i+1}")
        lastState_blue, lastAction_blue, lastReward_blue, lastState_red, lastAction_red, lastReward_red = reset_state(lastState_blue, lastAction_blue, lastReward_blue, lastState_red, lastAction_red,
                    lastReward_red)
        env.reset(seed = 42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            state = obtain_state(observation, agent)
            #     action = random_action(observation)
            if termination or truncation:
                print (reward)
                print (agent)
                print("termination or truncation:")
                if agent == "player_2":
                    # blue is x -- player_1, red is o -- player_2
                    lastAction_blue = action['pos']
                    lastReward_red = reward
                    lastReward_blue = -1 * reward
                else:
                    lastAction_red = action['pos']
                    lastReward_blue = reward
                    lastReward_red = -1 * reward
                for one in ['player_1', 'player_2']:
                    S = lastState_blue if one == 'player_1' else lastState_red
                    a = lastAction_blue if one == 'player_1' else lastAction_red
                    R = lastReward_blue if one == 'player_1' else lastReward_red
                    print('lastState S:\n', S)
                    print('lastAction a: ', a)
                    print('lastReward R: ', R)
                    maxQ_S_a = 0
                    Q_table[str(S)][str(a)] = (1 - ALPHA) * Q_table[str(S)][str(a)] \
                                              + ALPHA * (R + GAMMA * maxQ_S_a)
                    print('Q(S,a) = ', Q_table[str(S)][str(a)])
                action = None
                env.step(action)
                break
            else:
                print(agent)
                print(state)
                Q_table = updateQtable(agent, termination, truncation, observation, env, state, lastState_blue,
                                       lastState_red, lastAction_blue, lastAction_red, lastReward_blue, lastReward_red,
                                       Q_table, ALPHA, GAMMA)
                if agent == "player_1":
                    lastState_blue = state
                else:
                    lastState_red = state
                action = epsilon_greedy(Q_table, state, EPSILON, agent)
                if agent == "player_1":
                    lastAction_blue = action['pos']
                    lastReward_blue = reward
                else:
                    lastAction_red = action['pos']
                    lastReward_red = -1 * reward

                env.step(int(action['pos']))

        env.render()
    # print(Q_table)
    key_mapping = {'0': '(0,0)', '1': '(1,0)', '2': '(2,0)', '3': '(0,1)', '4': '(1,1)', '5': '(2,1)', '6': '(0,2)',
                   '7': '(1,2)', '8': '(2,2)'}
    new_Q_table = {}
    for state, actions in Q_table.items():
        new_actions = {(key_mapping[action]): value for action, value in actions.items()}
        new_Q_table[state] = new_actions
    print(new_Q_table)
    print('dim of state: ', len(new_Q_table))
    # with open('Q_table_dict.pkl', 'wb') as f:
    #     pickle.dump(new_Q_table, f)
if __name__ == "__main__":
    main()