import random
import numpy as np

class PDWorld:
    def __init__(self):
        self.world = {}
        self.init_world()
        self.agent_positions = {"red": (3, 3), "blue": (5, 3), "black": (1, 3)}
        self.agent_blocks = {"red": 0, "blue": 0, "black": 0}
        self.reward = {"red": 0, "blue": 0, "black": 0}

    def init_world(self):
        self.block_cells = [(1, 1), (3, 1), (4, 5), (1, 5), (2, 4), (5, 2)]
        for x in range(1, 6):
            for y in range(1, 6):
                self.world[(x, y)] = 0  # 0 represents empty space

        for cell in self.block_cells:
            self.world[cell] = 5  # Initial block count for each cell

    def get_state(self):
        red_pos = self.agent_positions["red"]
        blue_pos = self.agent_positions["blue"]
        black_pos = self.agent_positions["black"]
        red_block = 1 if self.agent_blocks["red"] > 0 else 0
        blue_block = 1 if self.agent_blocks["blue"] > 0 else 0
        black_block = 1 if self.agent_blocks["black"] > 0 else 0
        blocks = tuple(self.world[cell] for cell in self.block_cells)
        return red_pos + blue_pos + black_pos + (red_block, blue_block, black_block) + blocks

    def is_valid_move(self, agent, direction):
        x, y = self.agent_positions[agent]
        new_x, new_y = x + (direction == "east") - (direction == "west"), y + (direction == "north") - (direction == "south")
        new_pos = (new_x, new_y)
        return 1 <= new_x <= 5 and 1 <= new_y <= 5 and new_pos not in self.agent_positions.values()

    def move_agent(self, agent, direction):
        if self.is_valid_move(agent, direction):
            x, y = self.agent_positions[agent]
            new_x, new_y = x + (direction == "east") - (direction == "west"), y + (direction == "north") - (direction == "south")
            self.agent_positions[agent] = (new_x, new_y)

    def can_pickup(self, agent):
        x, y = self.agent_positions[agent]
        return self.world[(x, y)] > 0 and self.agent_blocks[agent] == 0

    def pickup(self, agent):
        if self.can_pickup(agent):
            x, y = self.agent_positions[agent]
            self.agent_blocks[agent] = self.world[(x, y)]
            self.world[(x, y)] = 0

    def can_dropoff(self, agent):
        x, y = self.agent_positions[agent]
        return self.agent_blocks[agent] > 0 and self.world[(x, y)] < 5

    def dropoff(self, agent):
        if self.can_dropoff(agent):
            x, y = self.agent_positions[agent]
            self.world[(x, y)] += self.agent_blocks[agent]
            self.agent_blocks[agent] = 0
        
    def prandom(self, agent):
        # Check if pickup and dropoff are applicable
        pickup_applicable = self.can_pickup(agent)
        dropoff_applicable = self.can_dropoff(agent)

        if pickup_applicable and dropoff_applicable:
            # Choose pickup or dropoff randomly
            action = random.choice(["pickup", "dropoff"])
        elif pickup_applicable:
            action = "pickup"
        elif dropoff_applicable:
            action = "dropoff"
        else:
            # If neither pickup nor dropoff is applicable, choose random action
            action = random.choice(["norht", "south", "east", "west"])  # Adjust choices as needed

        #in here 

        return action


    def pexploit(self, agent, q_values):
        # Check if pickup and dropoff are applicable
        pickup_applicable = self.can_pickup(agent)
        dropoff_applicable = self.can_dropoff(agent)

        if pickup_applicable and dropoff_applicable:
            return "pickup" if random.random() < 0.5 else "dropoff"
        elif pickup_applicable:
            return "pickup"
        elif dropoff_applicable:
            return "dropoff"
        else:
            # Apply the applicable operator with the highest q-value
            applicable_actions = ["move", "pickup", "dropoff"]
            max_q_value = max(q_values[action] for action in applicable_actions)
            best_actions = [action for action in applicable_actions if q_values[action] == max_q_value]
            # Break ties by rolling a dice for operators with the same utility
            return random.choice(best_actions)

    def pgreedy(self, agent, q_values):
        # Check if pickup and dropoff are applicable
        pickup_applicable = self.can_pickup(agent)
        dropoff_applicable = self.can_dropoff(agent)

        if pickup_applicable and dropoff_applicable:
            return "pickup" if random.random() < 0.5 else "dropoff"
        elif pickup_applicable:
            return "pickup"
        elif dropoff_applicable:
            return "dropoff"
        else:
            # Apply the applicable operator with the highest q-value
            applicable_actions = ["move", "pickup", "dropoff"]
            max_q_value = max(q_values[action] for action in applicable_actions)
            best_actions = [action for action in applicable_actions if q_values[action] == max_q_value]
            # Break ties by rolling a dice for operators with the same utility
            return random.choice(best_actions)

    def step(self, agent, qTableAgent, action=None, q_values=None, strategy=None):
        if action is None:
            # If action is not provided, use the specified strategy
            if strategy == "pexploit":
                action = self.pexploit(agent)
            elif strategy == "pgreedy":
                action = self.pgreedy(agent)
            elif strategy == "random":
                action = self.prandom(agent)
            else:
                raise ValueError("Invalid strategy")

        # Perform action based on the selected action
        if action == "pickup":
            self.pickup(agent)
        elif action == "dropoff":
            self.dropoff(agent)
        else:
            self.move_agent(agent, action)  # Adjust direction as needed
            # raise ValueError(f"Invalid action: {action}")

# Example usage:

world = PDWorld()
# grid size, grid size, num actions

q_table_red = np.zeros((5,5,6))
q_table_blue = np.zeros((5,5,6))
q_table_black = np.zeros((5,5,6))
# print(q_table)

# first 500 training loops (fills out qTable for each agent)
i = 0
while i < 500:
    world.step("red", q_table_red, None, None, "random")
    world.step("blue", q_table_blue, None, None, "random")
    world.step("black", q_table_black, None, None, "random")
    i+=1
# then need to do the 8500 execution loops
i = 0
while i < 8500:
    world.step("red", q_table_red, None, None, "pgreedy")
    world.step("blue", q_table_blue, None, None, "pgreedy")
    world.step("black", q_table_black, None, None, "pgreedy")
    i+=1

# # Perform actions
# world.step("red", "move")
# world.step("red", "move")
# world.step("red", "pickup")

# # Get current state
# current_state = world.get_state()
# print("Current state:", current_state)
