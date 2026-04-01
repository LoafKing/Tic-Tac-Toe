import pickle
import numpy as np
from pettingzoo.classic import tictactoe_v3

def obtain_state(observation, agent):
    board_state = observation['observation']
    A_1 = board_state[:, :, 0].transpose()
    A_2 = board_state[:, :, 1].transpose()
    state = A_1 - A_2 if agent == "player_1" else A_2 - A_1
    return state

def select_action_from_qtable(Q_table, state):
    state_key = str(state)
    if state_key not in Q_table:
        return None  # If state not in Q-table, return None
    action_values = Q_table[state_key]

    # Extract the best action as a tuple and convert it back to an integer index
    best_action = max(action_values, key=action_values.get)  # Action with highest Q-value
    try:
        best_action = int(best_action)  # If action is already an integer
    except ValueError:
        # Convert string representation of tuple (e.g., '(2,0)') to integer index
        action_map = {'(0,0)': 0, '(1,0)': 1, '(2,0)': 2, '(0,1)': 3, '(1,1)': 4,
                      '(2,1)': 5, '(0,2)': 6, '(1,2)': 7, '(2,2)': 8}
        best_action = action_map[best_action]

    return best_action

def test_trained_qtable(Q_table_path, episodes=10000):
    # Load the trained Q-table
    with open(Q_table_path, 'rb') as f:
        Q_table = pickle.load(f)

    env = tictactoe_v3.env(render_mode=None)  # Disable rendering for testing
    ai_wins = 0
    draws = 0
    total_games = episodes

    # Test the AI agent against a random opponent
    for episode in range(episodes):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            state = obtain_state(observation, agent)

            if termination or truncation:
                # Determine the winner
                if reward == -1 and agent == "player_2":  # AI agent wins
                    ai_wins += 1
                elif reward == 0:  # Draw
                    draws += 1
                break

            if agent == "player_1":
                # Player 1 uses the trained Q-table
                action = select_action_from_qtable(Q_table, state)
                if action is None:  # If state not in Q-table, choose a random action
                    action = np.random.choice(np.where(observation['action_mask'] == 1)[0])
            else:
                # Player 2 takes a random action
                action = np.random.choice(np.where(observation['action_mask'] == 1)[0])

            env.step(action)

    # Calculate win rate
    ai_win_rate = ai_wins / total_games
    draw_rate = draws / total_games
    loss_rate = 1 - ai_win_rate - draw_rate

    print(f"AI Win Rate: {ai_win_rate * 100:.2f}%")
    print(f"Draw Rate: {draw_rate * 100:.2f}%")
    print(f"Loss Rate: {loss_rate * 100:.2f}%")

    return ai_win_rate, draw_rate, loss_rate

# Path to the trained Q-table
Q_table_path = 'Q_table_dict.pkl'

# Test the Q-table for 10000 episodes
ai_win_rate, draw_rate, loss_rate = test_trained_qtable(Q_table_path, episodes=10000)