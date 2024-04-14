import numpy as np
import matplotlib.pyplot as plt

# Define the Agent class
class Agent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        # Initialize the Agent with default or provided parameters
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        # Initialize Q-table with zeros
        self.q_table = np.zeros((25, num_actions))

    # Method to choose action based on given policy
    def choose_action(self, state, policy):
        if policy == 'random':
            return self.choose_action_random(state)
        elif policy == 'exploit':
            return self.choose_action_exploit(state)
        elif policy == 'greedy':
            return self.choose_action_greedy(state)

    # Method to choose action randomly
    def choose_action_random(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    # Method to choose action exploiting the Q-table
    def choose_action_exploit(self, state):
        if np.random.rand() < self.exploration_rate:
            if np.random.rand() < 0.8:
                return np.argmax(self.q_table[state])
            else:
                return np.random.choice(np.where(self.q_table[state] == np.max(self.q_table[state]))[0])
        else:
            return np.argmax(self.q_table[state])

    # Method to choose greedy action
    def choose_action_greedy(self, state):
        return np.argmax(self.q_table[state])

    # Method to update Q-table based on the observed transition
    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - old_q_value)
        self.q_table[state, action] = new_q_value

    # Method to decay exploration rate
    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

# Define the SARSA Agent class, inheriting from Agent
class SARSA_Agent(Agent):
    # Override the update_q_table method for SARSA
    def update_q_table(self, state, action, reward, next_state, next_action=None):
        if next_action is None:
            raise ValueError("Next action must be provided for SARSA")
        old_q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value - old_q_value)
        self.q_table[state, action] = new_q_value

    # Override the choose_action method for SARSA
    def choose_action(self, state, policy):
        if policy == 'random':
            return self.choose_action_random(state)
        elif policy == 'exploit':
            return self.choose_action_exploit(state)
        elif policy == 'greedy':
            return self.choose_action_greedy(state)
        elif policy == 'SARSA':
            return self.choose_action_sarsa(state)

    # Method to choose action using SARSA strategy
    def choose_action_sarsa(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

# Define the Environment class
class Environment:
    def __init__(self):
        # Initialize environment parameters
        self.grid_size = 5
        self.num_agents = 3
        self.pickup_locations = [(1, 5), (2, 4), (5, 2)]  
        self.dropoff_locations = [(1, 1), (3, 1), (4, 5)] 
        self.agent_locations = [(3, 3), (5, 3), (1, 3)]
        self.agent_colors = ['red', 'blue', 'black']  
        self.agent_blocks = [0, 0, 0]
        self.blocks_at_pickup = [5, 5, 5]
        self.max_blocks_at_dropoff = 5
        self.paths = [[] for _ in range(self.num_agents)]
        
        # Validate pickup and drop-off locations
        self.validate_locations(self.pickup_locations)
        self.validate_locations(self.dropoff_locations)
        
    # Method to validate locations
    def validate_locations(self, locations):
        for loc in locations:
            if not (1 <= loc[0] <= self.grid_size) or not (1 <= loc[1] <= self.grid_size):
                raise ValueError("Location out of grid bounds")

    # Method to reset environment
    def reset(self):
        self.agent_locations = [(3, 3), (5, 3), (1, 3)]
        self.agent_blocks = [0, 0, 0]
        self.blocks_at_pickup = [5, 5, 5]
        self.paths = [[] for _ in range(self.num_agents)]

    # Method to get state for a given agent
    def get_state(self, agent_id):
        agent_loc = self.agent_locations[agent_id]
        return (agent_loc[0] - 1) * (self.grid_size - 1) + (agent_loc[1] - 1)  

    # Method to move agent and compute reward
    def move_agent(self, agent_id, action):
        # Move agent
        current_loc = self.agent_locations[agent_id]
        new_loc = current_loc
        # Update location based on action
        if action == 0:  
            new_loc = (max(current_loc[0] - 1, 1), current_loc[1])
        elif action == 1:  
            new_loc = (min(current_loc[0] + 1, self.grid_size), current_loc[1])
        elif action == 2:  
            new_loc = (current_loc[0], max(current_loc[1] - 1, 1))
        elif action == 3:  
            new_loc = (current_loc[0], min(current_loc[1] + 1, self.grid_size))
        # Prevent collisions
        if new_loc in self.agent_locations:
            new_loc = current_loc
        # Update agent location
        self.agent_locations[agent_id] = new_loc
        self.paths[agent_id].append(new_loc)
        # Compute reward
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
    
    # Method to compute Manhattan distance between two locations
    def manhattan_distance(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    # Method to get average Manhattan distance between agents and drop-off locations
    def get_avg_manhattan_distance(self):
        total_distance = 0
        for agent_id in range(self.num_agents):
            agent_loc = self.agent_locations[agent_id]
            dropoff_loc = self.dropoff_locations[agent_id]
            total_distance += self.manhattan_distance(agent_loc, dropoff_loc)
        return total_distance / self.num_agents

# Define a function to train agents
def train_agents(env, agents, num_iterations, policy):
    for _ in range(2):  
        for iteration in range(num_iterations):
            env.reset()  
            for agent_id in range(env.num_agents):
                agent = agents[agent_id]
                state = env.get_state(agent_id)
                total_reward = 0
                action = None  
                for _ in range(100):  
                    if policy == 'SARSA':
                        next_action = agent.choose_action(state, policy)  
                    action = agent.choose_action(state, policy)  
                    reward = env.move_agent(agent_id, action)
                    next_state = env.get_state(agent_id)
                    total_reward += reward
                    if policy == 'SARSA':
                        agent.update_q_table(state, action, reward, next_state, next_action)  
                    else:
                        agent.update_q_table(state, action, reward, next_state)
                    state = next_state
                    if total_reward > 0:  
                        break
                agent.decay_exploration_rate()

# Define a function to plot Q-table heatmap
def plot_q_table_heatmap(q_table, action_labels, vmin=None, vmax=None, state_labels=None, action_ticks=False):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(q_table, cmap='viridis',
                    aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Q-Value')
    plt.xticks(np.arange(len(action_labels)), action_labels,
               rotation=45) if action_labels else plt.xticks([])
    plt.xlabel('Actions')
    if state_labels:
        plt.yticks(np.arange(len(state_labels)), state_labels)
    plt.ylabel('States')
    if action_ticks and action_labels:
        plt.xticks(np.arange(len(action_labels)), action_labels)
    num_states, num_actions = q_table.shape
    for i in range(num_states + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(num_actions + 1):
        plt.axvline(j - 0.5, color='black', linewidth=0.5)
    plt.title('Q-Table Heatmap')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Define experiment 1
def experiment_1(env, agents):
    action_labels = ['North', 'South', 'West', 'East']
    print("Experiment 1:")
    train_agents(env, agents, num_iterations=500, policy='random')  
    train_agents(env, agents, num_iterations=8500, policy='random')
    train_agents(env, agents, num_iterations=8500, policy='greedy')
    train_agents(env, agents, num_iterations=8500, policy='exploit')

    print("Q-table for exploit policy:")
    for agent_id, agent in enumerate(agents):
        agent_color = env.agent_colors[agent_id]  
        print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
        print(agent.q_table)
        plot_q_table_heatmap(agent.q_table, action_labels)
        
    avg_distance = env.get_avg_manhattan_distance()
    print("Average Manhattan Distance:", avg_distance)

# Define experiment 2
def experiment_2(env, q_learning_agents, sarsa_agents):
    action_labels = ['North', 'South', 'West', 'East']
    print("Experiment 2 with Q-learning:")
    train_agents(env, q_learning_agents, num_iterations=9000, policy='greedy')

    print("Experiment 2 with SARSA:")
    train_agents(env, sarsa_agents, num_iterations=9000, policy='SARSA')

    print("Q-table for SARSA policy:")
    for agent_id, agent in enumerate(sarsa_agents):
        agent_color = env.agent_colors[agent_id]  
        print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
        print(agent.q_table)
        plot_q_table_heatmap(agent.q_table, action_labels)
        
    avg_distance = env.get_avg_manhattan_distance()
    print("Average Manhattan Distance:", avg_distance)

# Define experiment 3
def experiment_3(env, learning_rates):
    action_labels = ['North', 'South', 'West', 'East']
    for i, alpha in enumerate(learning_rates):
        print(f"Experiment 3 with Q-learning (Learning Rate: {alpha}):")
        q_learning_agents = [Agent(num_actions=4, learning_rate=alpha, discount_factor=0.5) for _ in range(env.num_agents)]
        train_agents(env, q_learning_agents, num_iterations=9000, policy='exploit')
        for agent_id, agent in enumerate(q_learning_agents):
            agent_color = env.agent_colors[agent_id]
            print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
            print(agent.q_table)
            plot_q_table_heatmap(agent.q_table, action_labels)
            
    avg_distance = env.get_avg_manhattan_distance()
    print("Average Manhattan Distance:", avg_distance)

    for i, alpha in enumerate(learning_rates):
        print(f"Experiment 3 with SARSA (Learning Rate: {alpha}):")
        sarsa_agents = [SARSA_Agent(num_actions=4, learning_rate=alpha, discount_factor=0.5) for _ in range(env.num_agents)]
        train_agents(env, sarsa_agents, num_iterations=9000, policy='SARSA')
        for agent_id, agent in enumerate(sarsa_agents):
            agent_color = env.agent_colors[agent_id]
            print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
            print(agent.q_table)
            plot_q_table_heatmap(agent.q_table, action_labels)
        
    avg_distance = env.get_avg_manhattan_distance()
    print("Average Manhattan Distance:", avg_distance)

# Define experiment 4
def experiment_4(env, agents):
    action_labels = ['North', 'South', 'West', 'East']
    print("Experiment 4:")
    train_agents_exp4(env, agents, num_iterations=10000, policy='random')  
    train_agents_exp4(env, agents, num_iterations=15000, policy='exploit')  

    print("Q-tables after Experiment 4:")
    for agent_id, agent in enumerate(agents):
        agent_color = env.agent_colors[agent_id]  
        print(f"Agent {agent_id} (Color: {agent_color}) Q-table:")
        print(agent.q_table)
        plot_q_table_heatmap(agent.q_table, action_labels)
        
    avg_distance = env.get_avg_manhattan_distance()
    print("Average Manhattan Distance:", avg_distance)

# Define training method for experiment 4
def train_agents_exp4(env, agents, num_iterations, policy):
    terminal_state_count = 0  
    for iteration in range(num_iterations):
        env.reset()  
        for agent_id in range(env.num_agents):
            agent = agents[agent_id]
            state = env.get_state(agent_id)
            total_reward = 0
            action = None  
            for _ in range(100):  
                if policy == 'SARSA':
                    next_action = agent.choose_action(state, policy)  
                action = agent.choose_action(state, policy)  
                reward = env.move_agent(agent_id, action)
                next_state = env.get_state(agent_id)
                total_reward += reward
                if policy == 'SARSA':
                    agent.update_q_table(state, action, reward, next_state, next_action)  
                else:
                    agent.update_q_table(state, action, reward, next_state)
                state = next_state
                if total_reward > 0:  
                    break
            agent.decay_exploration_rate()
            if total_reward > 0:
                terminal_state_count += 1
                if terminal_state_count == 3:
                    env.pickup_locations = [(4, 2), (3, 3), (2, 4)]
            if terminal_state_count == 6:
                break

# Define the main function to execute experiments
def main():
    env = Environment()
    agents = [Agent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]
    q_learning_agents = [Agent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]
    sarsa_agents = [SARSA_Agent(num_actions=4, learning_rate=0.3, discount_factor=0.5) for _ in range(env.num_agents)]
    learning_rates = [0.15, 0.45]

    experiment_1(env, agents)
    experiment_2(env, q_learning_agents, sarsa_agents)
    experiment_3(env, learning_rates)
    experiment_4(env, agents)

# Execute main function if this script is run directly
if __name__ == "__main__":
    main()
