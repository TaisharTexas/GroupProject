import numpy as np

# Define the parameters
GRID_SIZE = 5
NUM_PICKUP_LOCATIONS = 3
NUM_DROPOFF_LOCATIONS = 3
MAX_BLOCKS_PICKUP = 5
MAX_BLOCKS_DROPOFF = 5
AGENT_CAPACITY = 1
REWARD_PICKUP_DROPOFF = 13
REWARD_TRAVEL = -1

# Initialize the Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, AGENT_CAPACITY + 1, 4))  # 4 actions: up, down, left, right

# Define the action policy
def random_action():
    return np.random.randint(0, 4)  # 0: up, 1: down, 2: left, 3: right

# Define the state transitions
def move_agent(state, action):
    x, y, blocks = state
    if action == 0:  # up
        x = max(x - 1, 0)
    elif action == 1:  # down
        x = min(x + 1, GRID_SIZE - 1)
    elif action == 2:  # left
        y = max(y - 1, 0)
    elif action == 3:  # right
        y = min(y + 1, GRID_SIZE - 1)
    return (x, y, blocks)

# Define the main training loop
def train(num_episodes, learning_rate, discount_factor, exploration_rate):
    for episode in range(num_episodes):
        state = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE), 0)  # Random start state
        while True:
            action = random_action()  # Random action
            next_state = move_agent(state, action)
            
            # Calculate rewards
            reward = REWARD_TRAVEL
            if next_state[:2] in pickup_locations:
                reward = REWARD_PICKUP_DROPOFF
            elif next_state[:2] in dropoff_locations:
                reward = REWARD_PICKUP_DROPOFF
            
            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += learning_rate * (reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])
            
            state = next_state
            
            if reward == REWARD_PICKUP_DROPOFF:
                break

def print_q_table(q_table):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(AGENT_CAPACITY + 1):
                print(f"State: ({i}, {j}, {k})")
                for action, action_name in enumerate(['up', 'down', 'left', 'right']):
                    print(f"Action: {action_name} - Q-value: {q_table[i][j][k][action]}")
                print("")   

# Example usage
pickup_locations = [(1, 1), (3, 2), (4, 4)]
dropoff_locations = [(0, 3), (2, 0), (4, 2)]
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1

train(num_episodes, learning_rate, discount_factor, exploration_rate)
print_q_table(q_table)
