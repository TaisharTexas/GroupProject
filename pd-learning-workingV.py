# Define the state space
states = ['1', '2', '3', '4', "1'", "2'", "3'", "4'"]

# Define the actions
actions = {'p', 'w', 'e', 'd', 'n', 's'}

# Initialize the Q-table with zeros
Q_table = {}
for state in states:
    for action in actions:
        if state == '1' and action == 'e':
            Q_table[(state, action)] = 0
        elif state == '2' and action == 'w':
            Q_table[(state, action)] = 0
        elif state == '2' and action == 's':
            Q_table[(state, action)] = 0
        elif state == '3' and action == 'w':
            Q_table[(state, action)] = 0
        elif state == '4' and action == 'p':
            Q_table[(state, action)] = 0
        elif state == '4' and action == 'e':
            Q_table[(state, action)] = 0
        elif state == '4' and action == 'n':
            Q_table[(state, action)] = 0
        elif state == "1'" and action == 'e':
            Q_table[(state, action)] = 0
        elif state == "2'" and action == 'w':
            Q_table[(state, action)] = 0
        elif state == "2'" and action == 's':
            Q_table[(state, action)] = 0
        elif state == "3'" and action == 'w':
            Q_table[(state, action)] = 0
        elif state == "3'" and action == 'd':
            Q_table[(state, action)] = 0
        elif state == "4'" and action == 'e':
            Q_table[(state, action)] = 0
        elif state == "4'" and action == 'n':
            Q_table[(state, action)] = 0
        else:
            Q_table[(state, action)] = None
print("reached here")
print(Q_table)

# Define the policy
policy = ['w', 'p', 'e', 'd', 'w', 'p', 'e', 'd', 'w', 'p', 'n', 'e', 'w', 'e', 's', 'd', 'w', 'p']

# Define rewards
pickup_dropoff_reward = 13
move_penalty = 1

# Define learning parameters
alpha = 0.4  # Learning rate
gamma = 1.0  # Discount factor

# Define transition function
def transition(state, action):
    if state == '1' and action == 'e':
        return '2'
    elif state == "1'" and action == 'e':
        return "2'"
    elif state == '2' and action == 's':
        return '3'
    elif state == '2' and action == 'w':
        return '1'
    elif state == "2'" and action == 's':
        return "3'"
    elif state == "2'" and action == 'w':
        return "1'"
    elif state == '3' and action == 'w':
        return '4'
    elif state == "3'" and action == 'w':
        return "4'"
    elif state == "3'" and action == 'd':
        return '3'
    elif state == '4' and action == 'n':
        return '1'
    elif state == '4' and action == 'e':
        return '3'
    elif state == '4' and action == 'p':
        return "4'"
    elif state == "4'" and action == 'n':
        return "1'"
    elif state == "4'" and action == 'e':
        return "3'"
    else:
        return state

# Perform Q-learning
current_state = '3'
for action in policy:
    next_state = transition(current_state, action)
    if current_state == "3'" and action == 'd':
        reward = pickup_dropoff_reward
    elif current_state == '4' and action == 'p':
        reward = pickup_dropoff_reward
    else:
        reward = -move_penalty
    max_q_next = max(Q_table[(next_state, a)] for a in actions if Q_table[(next_state, a)] is not None)
    Q_table[(current_state, action)] += alpha * (reward + gamma * max_q_next - Q_table[(current_state, action)])
    current_state = next_state

# Print the updated Q-table
print("Updated Q-table:")
for key, value in Q_table.items():
    print(key[0], key[1], ", Q =", value)
