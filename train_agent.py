
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from env import DinoGame
import gymnasium as gym
import pygame


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Record the reward of the current step
        self.rewards.append(self.locals["rewards"])
        return True


gym.envs.registration.register(
    id='Game-v0',
    entry_point='__main__:DinoGame'
)



class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Log the reward for the current step
        reward = self.locals.get("rewards", [0])[0]  # Access the reward from the environment
        self.rewards.append(reward)
        return True

# Step 1: Initialize the environment
env = DinoGame()

# Step 2: Set up the DQN model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=100000, batch_size=32)

# Step 3: Define the number of timesteps
timesteps = 800000

# Step 4: Train the model
reward_callback = RewardLoggerCallback()
model.learn(total_timesteps=timesteps, callback=reward_callback)

# Step 5: Save the trained model
model_path = "game_model"
model.save(model_path)

# Optional: Check rewards logged during training
print("Rewards during training:")
print(reward_callback.rewards)



# Load the trained model
model = DQN.load("game_model")

# Initialize the environment
env = DinoGame()
obs, _ = env.reset()

frames = []

# Run the game through our learned policy
for _ in range(1000):  # Use 1000 steps instead of 1,000,000 for efficiency
    # Get our action from our learned policy
    action, _ = model.predict(obs, deterministic=True)

    # Take a step using the action
    obs, reward, done, truncated, info = env.step(action)

    # Render the current state to a Pygame surface and capture it as a frame
    screen = env.render()
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)  # Swap axes to match image dimensions
    frames.append(frame)

    if done:
        obs, _ = env.reset()

# Save the frames as a GIF
gif_path = "game.gif"
imageio.mimsave(gif_path, frames, fps=10)

# Display the GIF (if running in a Jupyter Notebook or Colab)
from IPython.display import Image
Image(filename=gif_path)