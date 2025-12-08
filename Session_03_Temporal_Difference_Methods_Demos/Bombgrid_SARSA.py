import numpy as np
from termcolor import colored
import random

# ===========================================================
#                    ENVIRONMENT CLASS
# ===========================================================
class BombEnv:
    """
    4x4 gridworld with:
      S (start), F (finish, +100), B (bomb, -100), P (plain, 0)
    Actions: 0=Left, 1=Right, 2=Up, 3=Down
    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self.state = 0
        self.forbidden_step = False
        
        self.reward_table = np.array([
            [0, -100, 0, 100],
            [0, 0, -100, 0],
            [0, 0, 0, 0],
            [-100, 0, 0, 0]
        ])
        
        self.done_table = np.array([
            [False, True, False, True],
            [False, False, True, False],
            [False, False, False, False],
            [True, False, False, False]
        ])
        
        self.render_table = np.array([
            ['S', 'B', 'P', 'F'],
            ['P', 'P', 'B', 'P'],
            ['P', 'P', 'P', 'P'],
            ['B', 'P', 'P', 'P']
        ])

    def reset(self):
        self.x, self.y = 0, 0
        self.state = 0
        self.forbidden_step = False
        return self.state

    def step(self, action):
        nx, ny = self.x, self.y
        
        if action == 0:    # LEFT
            nx = self.x - 1
        elif action == 1:  # RIGHT
            nx = self.x + 1
        elif action == 2:  # UP
            ny = self.y - 1
        elif action == 3:  # DOWN
            ny = self.y + 1
        
        # Boundary check
        if nx < 0 or nx > 3 or ny < 0 or ny > 3:
            self.forbidden_step = True
            nx, ny = self.x, self.y
        else:
            self.forbidden_step = False
            self.x, self.y = nx, ny
        
        self.state = 4 * self.x + self.y
        self.done = self.done_table[self.y, self.x]
        
        if self.forbidden_step:
            self.reward = -1
        else:
            self.reward = self.reward_table[self.y, self.x] - 1
        
        return self.state, self.reward, self.done

    def render(self):
        grid = self.render_table.copy().astype(object)
        grid[self.y, self.x] = colored('O', 'red')
        print('\n')
        for r in range(4):
            print(''.join(str(cell) for cell in grid[r]))


# ===========================================================
#                     SARSA AGENT (CORRECTED)
# ===========================================================
class SARSAgent:
    """
    SARSA: On-policy TD learning
    Update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    where a' is the ACTUAL next action chosen by the policy
    """
    def __init__(self, nr_states, nr_actions, alpha, gamma, epsilon, decay):
        print('Initial exploration rate:', epsilon)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.possibleActions = [0, 1, 2, 3]
        
        # FIX 1: Initialize to 0, not -100
        self.q_table = np.zeros((nr_states, nr_actions))

    def compute_action(self, state):
        """Epsilon-greedy action selection"""
        self.epsilon *= self.decay  # Decay epsilon
        
        if np.random.random() <= self.epsilon:
            action = random.choice(self.possibleActions)
        else:
            action = int(np.argmax(self.q_table[state, :]))
        
        return action

    def update_qtable(self, state, action, reward, new_state, next_action):
        """
        SARSA update using the ACTUAL next action.
        This is on-policy: we learn about the policy we're following.
        """
        # TD target using the next action we will actually take
        td_target = reward + self.gamma * self.q_table[new_state, next_action]
        td_error = td_target - self.q_table[state, action]
        
        # Update Q-value
        self.q_table[state, action] += self.alpha * td_error


# ===========================================================
#                     TRAINING PHASE
# ===========================================================

# Hyperparameters (IMPROVED)
alpha = 0.1
gamma = 0.9  # Higher discount (care more about future)
decay = 0.9999  # Even slower decay
epsilon = 1.0   # Start with full exploration
nr_states = 16
nr_actions = 4

# Create environment and agent
env = BombEnv()
agent = SARSAgent(nr_states, nr_actions, alpha, gamma, epsilon, decay)

# FIX 4: Optimistic initialization to encourage exploration
agent.q_table[:, :] = 10.0  # Start optimistic!

# FIX 5: More training episodes
print("\n=== TRAINING ===\n")
for i_episode in range(5000):  # Increased from 1000 to 5000
    state = env.reset()
    action = agent.compute_action(state)
    
    for t in range(200):
        new_state, reward, done = env.step(action)
        
        if done:
            # Terminal state
            agent.q_table[state, action] += agent.alpha * (reward - agent.q_table[state, action])
            break
        else:
            next_action = agent.compute_action(new_state)
            agent.update_qtable(state, action, reward, new_state, next_action)
            state = new_state
            action = next_action
    
    # Print progress every 1000 episodes
    if (i_episode + 1) % 1000 == 0:
        print(f"Episode {i_episode + 1}/5000 complete. Epsilon: {agent.epsilon:.4f}")

# Show learned Q-table
print("\n=== FINAL Q-TABLE ===")
print(agent.q_table)


# ===========================================================
#                     TESTING PHASE
# ===========================================================

print("\n=== TESTING (GREEDY POLICY) ===\n")
state = env.reset()
done = False
agent.epsilon = 0.0  # Pure exploitation
step_count = 0

env.render()

while not done and step_count < 20:  # Safety limit
    action = agent.compute_action(state)
    state, reward, done = env.step(action)
    step_count += 1
    env.render()

if step_count >= 20:
    print("\n⚠️ Agent got stuck (infinite loop)")
else:
    print(f"\n✅ Reached terminal state in {step_count} steps")