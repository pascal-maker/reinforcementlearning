import json
import sys

notebook_path = '/Users/pascal-maker/Desktop/reinforcementlearning/Session_05_Policy_Gradients_Assignment.ipynb'

# The new code with detailed comments
new_code = """def train_reinforce(env, agent, num_episodes=500, print_every=50, solved_reward=475):
    \"\"\"
    Train REINFORCE agent on environment.
    This function manages the main training loop over multiple episodes.
    \"\"\"
    reward_history = []
    loss_history = []
    # Deque to track the moving average of the last 100 episodes (for "solved" check)
    avg_rewards = deque(maxlen=100)
    for episode in range(num_episodes):
        # Step 0: Reset environment to start a new episode
        # We seed the reset for reproducibility
        state, info = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        # Steps 1-5: Collect episode trajectory (Monte Carlo)
        # REINFORCE is an episodic algorithm, so we must play until the end.
        while not done:
            # Step 1 & 2: Present state to Agent --> Agent selects Action
            # The agent uses its policy network to sample an action.
            action, _ = agent.select_action(state)
            # Step 3: Execute Action --> Get Reward & Next State
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Step 4: Store transition (s, a, r)
            # We need the full history to calculate discounted returns later.
            agent.store_transition(state, action, reward)
            # Step 5: Update state and accumulate total episode reward
            state = next_state
            episode_reward += reward
        # Step 6: Episode is done. Train on the stored transitions.
        # The agent calculates returns and performs a gradient update.
        loss = agent.train()
        # Track metrics for plotting/logging
        reward_history.append(episode_reward)
        loss_history.append(loss)
        avg_rewards.append(episode_reward)
        avg_reward = np.mean(avg_rewards)
        # Print progress every 'print_every' episodes
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.0f} | "
                  f"Avg(100): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")
        # Step 7: Check convergence / "Solved" condition
        # CartPole-v1 is considered solved if average reward > 475 over 100 episodes.
        if avg_reward >= solved_reward and len(avg_rewards) >= 100:
            print(f"\\n*** Solved in {episode + 1} episodes! ***")
            print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
            break
    return reward_history, loss_history
"""

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    found = False
    # Identify the target cell by its signature.
    # The original cell's source starts with the definition line.
    target_signature = "def train_reinforce(env, agent, num_episodes=500, print_every=50, solved_reward=475):"

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Check if this cell contains the function definition
            source_content = "".join(cell['source'])
            if target_signature in source_content:
                # We found the cell. Replace the source.
                # Format new_code into a list of strings with newlines
                new_source = []
                lines = new_code.splitlines(keepends=True)
                # Ensure all lines have \n except possibly the last one if it didn't in the input
                # splitlines(keepends=True) keeps the \n.
                cell['source'] = lines
                found = True
                print("Found and updated the target cell.")
                break
    
    if found:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print("Notebook file saved successfully.")
    else:
        print("Error: Target function definition not found in any cell.")
        sys.exit(1)

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
