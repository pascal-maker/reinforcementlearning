import gymnasium as gym
from gymnasium import wrappers
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import os


# Create video directory if it doesn't exist
os.makedirs('./video', exist_ok=True)

env = gym.make('CartPole-v1', render_mode='rgb_array')  # Use rgb_array for video recording
video_recorder = VideoRecorder(env, './video/episode.mp4')

for i in range(10):
    observation, info = env.reset()
    video_recorder.capture_frame()  # Capture initial frame
    
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        video_recorder.capture_frame()  # Capture frame after each step
        
        done = terminated or truncated
        if done:
            print(f"Episode {i+1} finished after {t+1} timesteps")
            break
    
video_recorder.close()
env.close()
