import numpy as np
from termcolor import colored
import random
import time

# ===========================================================
#                    ENVIRONMENT CLASS
# ===========================================================
class BombEnv:
    """
    Simple grid world environment:
    - S (start)
    - F (finish with +100 reward)
    - B (bomb with -100 reward)
    - P (plain/path with 0 reward)

    The agent moves in a 4x4 grid using 4 actions:
      0 = Left, 1 = Right, 2 = Up, 3 = Down
    """

    def __init__(self):
        # ---------------- Initialization ----------------
        # Start position: top-left corner
        self.x = 0
        self.y = 0
        self.state = 0
        self.forbidden_step = False  # Flag for invalid (out-of-bounds) moves

        # ---------------- Reward Table ----------------
        # Defines the reward for each cell in the grid.
        # Rows correspond to y, columns correspond to x.
        self.reward_table = np.array([
            [0, -100, 0, 100],   # Row 0: Start, Bomb, Plain, Finish
            [0, 0, -100, 0],     # Row 1
            [0, 0, 0, 0],        # Row 2
            [-100, 0, 0, 0]      # Row 3
        ])

        # ---------------- Done Table ----------------
        # Marks terminal states: finish or bomb cells (True = game over)
        self.done_table = np.array([
            [False, True, False, True],
            [False, False, True, False],
            [False, False, False, False],
            [True, False, False, False]
        ])

        # ---------------- Render Table ----------------
        # Used only for visualization in the terminal.
        # S = Start, F = Finish, B = Bomb, P = Plain.
        self.render_table = np.array([
            ['S', 'B', 'P', 'F'],
            ['P', 'P', 'B', 'P'],
            ['P', 'P', 'P', 'P'],
            ['B', 'P', 'P', 'P']
        ])

    def reset(self):
        """
        Reset environment to the start position (top-left corner).
        Called at the beginning of each episode.
        """
        self.x = 0
        self.y = 0
        return 0  # Return initial state index (for Q-table)

    def step(self, action):
        """
        Perform an action and update the environment.

        Parameters:
            action (int): 0=Left, 1=Right, 2=Up, 3=Down

        Returns:
            (new_state, reward, done)
        """

        # =========================================================
        #                     MOVE LOGIC
        # =========================================================
        if action == 0:  # LEFT
            if self.x - 1 < 0:
                self.forbidden_step = True  # Hit wall
            else:
                self.x -= 1
                self.forbidden_step = False

        elif action == 1:  # RIGHT
            if self.x + 1 > 3:
                self.forbidden_step = True
            else:
                self.x += 1
                self.forbidden_step = False

        elif action == 2:  # UP
            if self.y - 1 < 0:
                self.forbidden_step = True
            else:
                self.y -= 1
                self.forbidden_step = False

        elif action == 3:  # DOWN
            if self.y + 1 > 3:
                self.forbidden_step = True
            else:
                self.y += 1
                self.forbidden_step = False

        # =========================================================
        #                    STATE + REWARD UPDATE
        # =========================================================
        # Encode (x,y) into a single state index: state = 4*x + y
        self.state = 4 * self.x + self.y

        # Check if this is a terminal state (bomb or finish)
        self.done = self.done_table[self.y, self.x]

        # Assign reward based on cell
        if self.forbidden_step:
            self.reward = -1  # Penalty for trying to move outside grid
        else:
            # Reward from table minus a small step cost (-1)
            self.reward = self.reward_table[self.y, self.x] - 1

        return self.state, self.reward, self.done

    def render(self):
        """
        Print the 4x4 grid to the terminal.
        The agent's position is shown as a red 'O'.
        """
        # Copy grid to modify temporarily
        rendering = self.render_table.copy()

        # Place the agent marker
        rendering[self.y, self.x] = 'O'

        # Print grid line by line
        print('\n')
        for r in range(rendering.shape[0]):
            print('')
            for c in range(rendering.shape[1]):
                if rendering[r, c] == 'O':
                    print(colored('O', 'red'), end='')  # Agent
                else:
                    print(rendering[r, c], end='')      # Normal cell


# ===========================================================
#                     Q-LEARNING AGENT
# ===========================================================
class Qagent:
    """
    Implements a simple tabular Q-learning agent.
    - Learns a Q-table mapping (state, action) → expected reward.
    - Uses epsilon-greedy exploration strategy.
    """

    def __init__(self, nr_states, nr_actions, alpha, gamma, exploration_rate, decay):
        print('Initial exploration rate =', exploration_rate)

        # ---------------- Learning Parameters ----------------
        self.alpha = alpha             # Learning rate
        self.gamma = gamma             # Discount factor (future reward importance)
        self.exploration_rate = exploration_rate
        self.decay = decay             # Epsilon decay factor per action

        # ---------------- Environment Dimensions ----------------
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.possibleActions = [0, 1, 2, 3]  # L, R, U, D

        # ---------------- Q-Table Initialization ----------------
        # Start with low Q-values (-100) to encourage exploration initially.
        self.q_table = np.zeros((nr_states, nr_actions)) - 100

    def compute_action(self, state):
        """
        Choose the next action using epsilon-greedy exploration.

        - With probability ε: choose random action (exploration)
        - With probability (1−ε): choose best known action (exploitation)
        """
        # Gradually reduce epsilon each time an action is taken
        self.exploration_rate *= self.decay
        r = random.random()  # Random number between 0 and 1

        # ------------- Exploration branch -------------
        if r <= self.exploration_rate:
            action = random.choice(self.possibleActions)
            print('Exploration')

        # ------------- Exploitation branch -------------
        else:
            # Choose the action with the highest Q-value for this state
            action = np.argmax(self.q_table[state, :])
            print(r, self.exploration_rate, '→ Exploitation')

        # Store and return chosen action
        self.action = action
        return action

    def update_qtable(self, state, new_state, reward):
        """
        Q-learning update using Bellman equation.
        Update rule:
          Q(s,a) ← (1−α)Q(s,a) + α [ r + γ max_a' Q(s',a') ]
        """
        # Decide next Q-value target
        if random.random() <= self.exploration_rate:
            # Occasionally sample a random action value for exploration
            next_q_value = self.q_table[new_state, random.choice(self.possibleActions)]
        else:
            # Normally take best action's value (greedy)
            next_q_value = self.q_table[new_state, np.argmax(self.q_table[new_state, :])]

        # Apply Bellman equation to update current state-action pair
        self.q_table[state, self.action] = (
            (1 - self.alpha) * self.q_table[state, self.action] +
            self.alpha * (reward + self.gamma * next_q_value)
        )


# ===========================================================
#                     TRAINING PHASE
# ===========================================================

# --- Hyperparameters ---
alpha = 0.1              # Learning rate
gamma = 0.8              # Discount factor
decay = 0.9998           # Epsilon decay per step
exploration_rate = 0.999 # Starting epsilon
nr_states = 16           # 4x4 grid → 16 states
nr_actions = 4           # 4 possible moves

# --- Create environment + agent ---
env = BombEnv()
q_agent = Qagent(nr_states, nr_actions, alpha, gamma, exploration_rate, decay)

# --- Run training for 1000 episodes ---
for i_episode in range(1000):
    state = env.reset()  # Reset environment each episode
    print('Initial state =', state)

    # Up to 200 moves per episode (safety limit)
    for t in range(200):
        # Choose action (ε-greedy)
        action = q_agent.compute_action(state)
        print('Action chosen =', action)

        # Apply the action in the environment
        new_state, reward, done = env.step(action)
        print('Transition: state', state, '→', new_state)

        # Update Q-table only if move was valid
        if new_state != state:
            q_agent.update_qtable(state, new_state, reward)

        # Show the grid with the agent
        env.render()
        print('\nNew state =', new_state)

        # Stop episode if finish or bomb reached
        if done:
            break

# --- Show learned Q-values after training ---
print("\nFinal Q-table:")
print(q_agent.q_table)


# ===========================================================
#                     TESTING PHASE
# ===========================================================
done = False
state = env.reset()
env.render()
stopcounter = 0

# Run greedy policy (no random moves)
while not done:
    q_agent.exploration_rate = 0  # Force pure exploitation
    action = q_agent.compute_action(state)  # Always pick best action
    new_state, reward, done = env.step(action)
    state = new_state
    stopcounter += 1

    # Safety stop in case of infinite loop
    if stopcounter > 25:
        done = True

    # Render grid every step
    env.render()
