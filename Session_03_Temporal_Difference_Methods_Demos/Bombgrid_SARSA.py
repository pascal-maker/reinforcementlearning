import numpy as np
from termcolor import colored
import random
import time

# ---------------------- ENVIRONMENT CLASS ----------------------
class BombEnv:
    """
    Simple grid world environment:
    - S (start)
    - F (finish with +100 reward)
    - B (bomb with -100 reward)
    - P (path with 0 reward)
    The agent moves in a 4x4 grid using actions: left, right, up, down.
    """

    def __init__(self):
        # Starting position
        self.x = 0
        self.y = 0
        self.state = 0
        self.forbidden_step = False

        # Reward table for each grid cell
        self.reward_table = np.array([
            [0, -100, 0, 100],   # Row 0
            [0, 0, -100, 0],     # Row 1
            [0, 0, 0, 0],        # Row 2
            [-100, 0, 0, 0]      # Row 3
        ])

        # Table marking terminal (done) states (True = episode ends)
        self.done_table = np.array([
            [False, True, False, True],
            [False, False, True, False],
            [False, False, False, False],
            [True, False, False, False]
        ])

        # For visualization
        self.render_table = np.array([
            ['S', 'B', 'P', 'F'],
            ['P', 'P', 'B', 'P'],
            ['P', 'P', 'P', 'P'],
            ['B', 'P', 'P', 'P']
        ])

    def reset(self):
        """Reset environment to starting position (0,0)."""
        self.x = 0
        self.y = 0
        return 0  # Return initial state index

    def step(self, action):
        """
        Execute an action:
        0 = Left, 1 = Right, 2 = Up, 3 = Down
        Returns (new_state, reward, done)
        """

        # ---- MOVE LOGIC ----
        if action == 0:  # LEFT
            if self.x - 1 < 0:
                self.forbidden_step = True  # Tried to move out of bounds
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

        # ---- STATE UPDATE ----
        self.state = 4 * self.x + self.y
        self.done = self.done_table[self.y, self.x]

        # ---- REWARD ----
        if self.forbidden_step:
            self.reward = -1  # Small penalty for hitting wall
        else:
            # Reward from table minus a small step cost (-1)
            self.reward = self.reward_table[self.y, self.x] - 1

        return self.state, self.reward, self.done

    def render(self):
        """Visualize the grid with the agent's position marked in red."""
        rendering = self.render_table.copy()
        rendering[self.y, self.x] = 'O'  # Mark current position

        print('\n')
        for r in range(rendering.shape[0]):
            print('')
            for c in range(rendering.shape[1]):
                if rendering[r, c] == 'O':
                    print(colored('O', 'red'), end='')
                else:
                    print(rendering[r, c], end='')

# ---------------------- Q-LEARNING AGENT ----------------------
class Qagent:
    def __init__(self, nr_states, nr_actions, alpha, gamma, exploration_rate, decay):
        print('Initial exploration rate =', exploration_rate)
        self.alpha = alpha                    # Learning rate
        self.gamma = gamma                    # Discount factor
        self.exploration_rate = exploration_rate
        self.decay = decay                    # Decay of epsilon
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.possibleActions = [0, 1, 2, 3]   # 4 possible moves
        self.q_table = np.zeros((nr_states, nr_actions)) - 100  # Init low Q-values

    def compute_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        self.exploration_rate *= self.decay  # Gradually reduce exploration
        r = random.random()

        # ---- Exploration ----
        if r <= self.exploration_rate:
            action = random.choice(self.possibleActions)
            print('Exploration')
        # ---- Exploitation ----
        else:
            action = np.argmax(self.q_table[state, :])
            print(r, self.exploration_rate, '→ Exploitation')

        self.action = action
        return action

    def update_qtable(self, state, new_state, reward):
        """Update Q-table based on reward and future expected value."""
        if random.random() <= self.exploration_rate:
            next_q_value = self.q_table[new_state, random.choice(self.possibleActions)]
        else:
            next_q_value = self.q_table[new_state, np.argmax(self.q_table[new_state, :])]

        # Bellman equation update
        self.q_table[state, self.action] = (1 - self.alpha) * self.q_table[state, self.action] + \
                                           self.alpha * (reward + self.gamma * next_q_value)

# ---------------------- TRAINING LOOP ----------------------
# Agent hyperparameters
alpha = 0.1
gamma = 0.8
decay = 0.9998
exploration_rate = 0.999
nr_states = 16
nr_actions = 4

# Initialize environment and agent
env = BombEnv()
q_agent = Qagent(nr_states, nr_actions, alpha, gamma, exploration_rate, decay)

# Train for 1000 episodes
for i_episode in range(1000):
    state = env.reset()
    print('Initial state =', state)
    for t in range(200):  # max steps per episode
        action = q_agent.compute_action(state)
        print('Action chosen =', action)
        new_state, reward, done = env.step(action)
        print('Transition: state', state, '→', new_state)
        if new_state != state:
            q_agent.update_qtable(state, new_state, reward)

        env.render()
        print('\nNew state =', new_state)

        # End if goal or bomb reached
        if done:
            break

# Print final learned Q-table
print("\nFinal Q-table:")
print(q_agent.q_table)

# ---------------------- TESTING ----------------------
done = False
state = env.reset()
env.render()
stopcounter = 0

# Run greedy policy (no exploration)
while not done:
    q_agent.exploration_rate = 0
    action = q_agent.compute_action(state)
    new_state, reward, done = env.step(action)
    state = new_state
    stopcounter += 1
    if stopcounter > 25:  # Avoid infinite loops
        done = True
    env.render()
