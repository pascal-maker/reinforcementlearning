import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode='human')

# Reset and get initial state
state, info = env.reset()
done = False
angle = state[2]

while not done:
    if angle < 0:
        # Push cart left
        state, reward, terminated, truncated, info = env.step(0)
    else:
        # Push cart right
        state, reward, terminated, truncated, info = env.step(1)

    done = terminated or truncated
    angle = state[2]
    env.render()

env.close()
