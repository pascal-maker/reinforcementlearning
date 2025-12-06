import numpy as np
from termcolor import colored
import random

# ------------ Gridworld Environment ------------
class BombEnv:
    """
    4x4 gridworld with:
      S (start), F (finish, +100), B (bomb, -100), P (plain, 0)
    Step cost: -1 on any valid move.
    Invalid move (hit wall): reward -1 and stay put.
    Terminal when at F or B.
    State id: state = 4*x + y. (Note: tables indexed as [y, x].)
    """

    def __init__(self):
        # Agent position (x, y). Start at top-left (0,0).
        self.x = 0
        self.y = 0

        # Encoded state id (column-major): state = 4*x + y
        self.state = 0

        # Flag set to True when the agent tries to move out of bounds
        self.forbidden_step = False

        # Rewards shown here are terminal bonuses; a step cost (-1) is added separately.
        # Layout comment shows y as rows and x as columns (y=0 top row).
        self.reward_table = np.array([
            [   0, -100,    0, 100],  # y=0, x=0..3
            [   0,    0, -100,   0],  # y=1
            [   0,    0,    0,   0],  # y=2
            [-100,    0,    0,   0],  # y=3
        ])

        # done_table marks terminal states (bombs and finish).
        self.done_table = np.array([
            [False,  True, False,  True],
            [False, False,  True, False],
            [False, False, False, False],
            [ True, False, False, False],
        ])

        # For pretty printing. S=start, F=finish, B=bomb, P=plain.
        self.render_table = np.array([
            ['S','B','P','F'],
            ['P','P','B','P'],
            ['P','P','P','P'],
            ['B','P','P','P'],
        ])

    def reset(self):
        """Reset to start S at (x=0,y=0). Return initial state id."""
        self.x, self.y = 0, 0          # go back to start
        self.state = 0                 # encoded as 4*0 + 0
        self.forbidden_step = False    # clear wall-bump flag
        return self.state

    def step(self, action):
        """
        Apply action (0=L, 1=R, 2=U, 3=D).
        Returns: (new_state, reward, done)
        """
        self.action = action  # store for debugging / learning update

        # Propose next coordinates based on action
        nx, ny = self.x, self.y
        if action == 0:      # LEFT
            nx = self.x - 1
        elif action == 1:    # RIGHT
            nx = self.x + 1
        elif action == 2:    # UP
            ny = self.y - 1
        elif action == 3:    # DOWN
            ny = self.y + 1

        # Bound check: if out of grid, stay put and mark as forbidden (wall bump)
        if nx < 0 or nx > 3 or ny < 0 or ny > 3:
            self.forbidden_step = True
            nx, ny = self.x, self.y   # don't move
        else:
            self.forbidden_step = False
            self.x, self.y = nx, ny   # commit the move

        # Update encoded state id (column-major mapping)
        self.state = 4*self.x + self.y

        # Check if new cell is terminal (bomb or finish)
        self.done = self.done_table[self.y, self.x]

        # Reward: if wall bump → -1; else cell reward minus step cost
        if self.forbidden_step:
            self.reward = -1
        else:
            # reward table has terminal bonuses, we always subtract step cost (-1)
            self.reward = self.reward_table[self.y, self.x] - 1

        return self.state, self.reward, self.done

    def render(self):
        """Print the grid; agent shown as red 'O'."""
        grid = self.render_table.copy().astype(object)  # copy so we don't overwrite original
        grid[self.y, self.x] = colored('O', 'red')      # mark agent position

        print()  # blank line for spacing
        for r in range(4):
            # Join row elements without spaces (e.g., "SBPF")
            print(''.join(str(cell) for cell in grid[r]))


# ------------ Q-learning Agent ------------
class Qagent:
    """
    Tabular Q-learning with epsilon-greedy exploration.
    - epsilon decays multiplicatively each action: epsilon *= decay
    - Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, nr_states, nr_actions, alpha, gamma, epsilon, decay):
        # Hyperparameters
        print('exploration rate (initial):', epsilon)
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # exploration rate
        self.decay   = decay    # multiplicative decay per decision

        # Problem dimensions
        self.nr_states  = nr_states
        self.nr_actions = nr_actions

        # Initialize Q-table with zeros: shape [states, actions]
        self.q_table = np.zeros((nr_states, nr_actions))

        # Enumerate actions (0=L, 1=R, 2=U, 3=D)
        self.possibleActions = [0, 1, 2, 3]

        # Last chosen action (used by the update)
        self.action = 0

    def compute_action(self, state):
        """
        Epsilon-greedy action selection with multiplicative decay.
        With probability epsilon → explore (random action),
        else → exploit (argmax over Q[state]).
        """
        # Decay epsilon at every decision to reduce exploration over time
        self.epsilon *= self.decay

        # Draw uniform random number in [0,1]
        if np.random.uniform() <= self.epsilon:
            # Explore: choose a random legal action
            self.action = random.choice(self.possibleActions)
            # print('exploration')
        else:
            # Exploit: choose the greedy action (break ties by first argmax index)
            self.action = int(np.argmax(self.q_table[state, :]))
            # print('exploitation')

        return self.action

    def update_qtable(self, state, new_state, reward):
        """
        Off-policy Q-learning update:
          Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        """
        # Best action-value at the next state (bootstrap target)
        best_next = np.max(self.q_table[new_state, :])

        # TD target and TD error
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state, self.action]

        # Update Q(s,a) towards the target
        self.q_table[state, self.action] += self.alpha * td_error

        # NOTE (optional improvement):
        # If the environment signals terminal at new_state,
        # you can skip bootstrapping by setting td_target = reward instead.
        # That would require passing `done` into this function.


# ------------ Training loop ------------
# Hyperparameters for learning
nr_of_episodes = 1000  # total episodes to sample
alpha   = 0.1          # learning rate
gamma   = 0.8          # discount factor
decay   = 0.9999       # per-decision epsilon decay
epsilon = 0.999        # initial exploration rate

# Environment/action-space sizes
nr_states  = 16  # 4x4 grid
nr_actions = 4   # L, R, U, D

# Create environment and agent
env = BombEnv()
agent = Qagent(nr_states, nr_actions, alpha, gamma, epsilon, decay)

# Main training loop: interact → observe → learn
for i_episode in range(nr_of_episodes):
    state = env.reset()          # start new episode from S
    for t in range(200):         # safety cap on steps per episode
        action = agent.compute_action(state)          # policy: epsilon-greedy
        new_state, reward, done = env.step(action)    # env transition
        agent.update_qtable(state, new_state, reward) # learn from (s,a,r,s')
        state = new_state                              # move forward
        # env.render()  # (optional) visualize each step (slow)
        if done:
            break  # end episode on terminal state (bomb or finish)

# Show learned Q-values
print("Trained Q-table:\n", agent.q_table)

# ------------ Test the learned policy (greedy) ------------
# Evaluate without exploration to see the path the agent learned
state = env.reset()
done = False
agent.epsilon = 0.0  # pure greedy at test time

# Render initial grid
env.render()

# Step using greedy policy until terminal
while not done:
    action = agent.compute_action(state)   # with epsilon=0, this is argmax
    state, reward, done = env.step(action)
    env.render()                           # visualize current position
