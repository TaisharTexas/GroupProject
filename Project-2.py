import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((25, num_actions))

    def choose_action(self, state, policy):
        if policy == 'PRANDOM':
            return self.choose_action_prandom(state)
        elif policy == 'PEXPLOIT':
            return self.choose_action_pexploit(state)
        elif policy == 'PGREEDY':
            return self.choose_action_pgreedy(state)

    def choose_action_prandom(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def choose_action_pexploit(self, state):
        if np.random.rand() < self.exploration_rate:
            if np.random.rand() < 0.8:
                return np.argmax(self.q_table[state])
            else:
                return np.random.choice(np.where(self.q_table[state] == np.max(self.q_table[state]))[0])
        else:
            return np.argmax(self.q_table[state])

    def choose_action_pgreedy(self, state):
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - old_q_value)
        self.q_table[state, action] = new_q_value

    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
        
class SARSAQAgent(QLearningAgent):
    def update_q_table(self, state, action, reward, next_state, next_action=None):
        if next_action is None:
            raise ValueError("Next action must be provided for SARSA")
        old_q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value - old_q_value)
        self.q_table[state, action] = new_q_value


    def choose_action(self, state, policy):
        if policy == 'PRANDOM':
            return self.choose_action_prandom(state)
        elif policy == 'PEXPLOIT':
            return self.choose_action_pexploit(state)
        elif policy == 'PGREEDY':
            return self.choose_action_pgreedy(state)
        elif policy == 'SARSA':
            return self.choose_action_sarsa(state)

    def choose_action_sarsa(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

class Environment:
    def __init__(self):
        self.grid_size = 5
        self.num_agents = 3
        self.pickup_locations = [(1, 5), (2, 4), (5, 2)]  
        self.dropoff_locations = [(1, 1), (3, 1), (4, 5)] 
        self.agent_locations = [(3, 3), (5, 3), (1, 3)]
        self.agent_colors = ['red', 'blue', 'black']  # Assign colors to agents
        self.agent_blocks = [0, 0, 0]
        self.blocks_at_pickup = [5, 5, 5]
        self.max_blocks_at_dropoff = 5
        self.paths = [[] for _ in range(self.num_agents)]  # List to store paths taken by agents

    def reset(self):
        self.agent_locations = [(3, 3), (5, 3), (1, 3)]
        self.agent_blocks = [0, 0, 0]
        self.blocks_at_pickup = [5, 5, 5]
        self.paths = [[] for _ in range(self.num_agents)]

    def get_state(self, agent_id):
        agent_loc = self.agent_locations[agent_id]
        return (agent_loc[0] - 1) * (self.grid_size - 1) + (agent_loc[1] - 1)  

    def move_agent(self, agent_id, action):
        current_loc = self.agent_locations[agent_id]
        new_loc = current_loc

        if action == 0:  # Up
            new_loc = (max(current_loc[0] - 1, 1), current_loc[1])
        elif action == 1:  # Down
            new_loc = (min(current_loc[0] + 1, self.grid_size), current_loc[1])
        elif action == 2:  # Left
            new_loc = (current_loc[0], max(current_loc[1] - 1, 1))
        elif action == 3:  # Right
            new_loc = (current_loc[0], min(current_loc[1] + 1, self.grid_size))

        # Check for collision with other agents
        if new_loc in self.agent_locations:
            # If the new location is already occupied, stay in the current position
            new_loc = current_loc

        self.agent_locations[agent_id] = new_loc
        self.paths[agent_id].append(new_loc)  # Record the path taken by the agent

        reward = -1
        if new_loc in self.pickup_locations:
            pickup_index = self.pickup_locations.index(new_loc)
            if self.blocks_at_pickup[pickup_index] > 0 and self.agent_blocks[agent_id] == 0:
                reward += 13
                self.blocks_at_pickup[pickup_index] -= 1
                self.agent_blocks[agent_id] += 1
        elif new_loc in self.dropoff_locations:
            dropoff_index = self.dropoff_locations.index(new_loc)
            if self.agent_blocks[agent_id] > 0 and self.agent_blocks[agent_id] < self.max_blocks_at_dropoff:
                reward += 13
                self.agent_blocks[agent_id] -= 1

        return reward

def train_agents(env, agents, num_iterations, policy):
    for _ in range(2):  # Run each experiment twice
        for iteration in range(num_iterations):
            env.reset()  # Reset the PD world to initial state
            for agent_id in range(env.num_agents):
                agent = agents[agent_id]
                state = env.get_state(agent_id)
                total_reward = 0
                action = None  # Initialize action for SARSA

                for _ in range(100):  
                    if policy == 'SARSA':
                        next_action = agent.choose_action(state, policy)  # Get next action for SARSA
                    action = agent.choose_action(state, policy)  # Get action for Q-learning or SARSA
                    reward = env.move_agent(agent_id, action)
                    next_state = env.get_state(agent_id)
                    total_reward += reward
                    if policy == 'SARSA':
                        agent.update_q_table(state, action, reward, next_state, next_action)  # Provide next_action for SARSA
                    else:
                        agent.update_q_table(state, action, reward, next_state)
                    state = next_state
                    if total_reward > 0:  
                        break

                agent.decay_exploration_rate()

            if iteration % 100 == 0:
                print("Experiment:", _ + 1, "Iteration:", iteration)
    
def plot_q_table_heatmap(q_table):
    plt.figure(figsize=(10, 6))
    plt.imshow(q_table, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.title('Q-Table Heatmap')
    plt.show()

if __name__ == "__main__":
    env = Environment()
    agents = [QLearningAgent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]
    
    # Experiment 1
    print("Experiment 1:")
    train_agents(env, agents, num_iterations=500, policy='PRANDOM')  # Initial training with PRANDOM for 500 steps

    # Switch policy to PRANDOM and continue training for 8500 steps
    train_agents(env, agents, num_iterations=8500, policy='PRANDOM')

    # Switch policy to PGREEDY and continue training for 8500 steps
    train_agents(env, agents, num_iterations=8500, policy='PGREEDY')

    # Switch policy to PEXPLOIT and continue training for 8500 steps
    train_agents(env, agents, num_iterations=8500, policy='PEXPLOIT')

    # After training for "PEXPLOIT" policy
    print("Q-table for PEXPLOIT policy:")
    for agent_id, agent in enumerate(agents):
        agent_color = env.agent_colors[agent_id]  # Get the color of the current agent
        print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
        print(agent.q_table)
        plot_q_table_heatmap(agent.q_table)

    env = Environment()
    q_learning_agents = [QLearningAgent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]
    sarsa_agents = [SARSAQAgent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]

    # Experiment 2 with Q-learning
    print("Experiment 2 with Q-learning:")
    train_agents(env, q_learning_agents, num_iterations=9000, policy='PGREEDY')

    # Experiment 2 with SARSA
    print("Experiment 2 with SARSA:")
    train_agents(env, sarsa_agents, num_iterations=9000, policy='SARSA')

    # Printing Q-table for SARSA
    print("Q-table for SARSA policy:")
    for agent_id, agent in enumerate(sarsa_agents):
        agent_color = env.agent_colors[agent_id]  # Get the color of the current agent
        print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
        print(agent.q_table)
        plot_q_table_heatmap(agent.q_table)
