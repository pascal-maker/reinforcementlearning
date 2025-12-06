import gymnasium as gym

env = gym.make("CartPole-v1")#create environment
for i_episode in range(20):#for each episode
    observation, info = env.reset()#reset environment
    for t in range(100):#for each step
        env.render()#render environment will create a popup windows
        print(observation)#print observation
        action = env.action_space.sample()#random action
        observation, reward, terminated, truncated, info = env.step(action)#false episode  is not finished true episode is finished
        if terminated or truncated: #if episode is finished naturally or episode is ended external limit
            print(f"Episode finished after {t+1} timesteps")
            break 
env.close()


