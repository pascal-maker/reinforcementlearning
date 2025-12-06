import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

# Reset returns (observation, info) tuple
state, info = env.reset()
done = False
angle = state[2]
previous_angle = angle

while not done:
    # If angle is decreasing (pole falling left), push left
    if (angle - previous_angle) < 0:
        state, reward, terminated, truncated, info = env.step(0)
    else:
        # If angle is increasing (pole falling right), push right
        state, reward, terminated, truncated, info = env.step(1)
    
    done = terminated or truncated
    previous_angle = angle
    angle = state[2]
    env.render()

env.close()