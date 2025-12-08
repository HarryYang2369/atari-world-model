import os
import random
import numpy as np
import gym
from env import make_env   
import shutil



# --- Hyperparameters ---
MAX_FRAMES = 1000      # max number of steps per episode
MAX_EPISODES = 1000     # number of rollouts to generate
ENV_NAME = "ALE/Breakout-v5"
RENDER = False         # set True to see Breakout while recording

DIR_NAME = "record_breakout"
if os.path.exists(DIR_NAME):
    shutil.rmtree(DIR_NAME)
os.makedirs(DIR_NAME, exist_ok=True)


def main():
    """
    Generate rollouts from Atari Breakout for VAE training.

    Each episode is saved as a .npz file with:
        - obs: uint8 array of shape (T, 84, 84, 1)
        - action: uint8 array of shape (T,)
    """
    
    total_frames = 0

    # Create wrapped Breakout env (returns 84x84x1 uint8 frames)
    env = make_env(ENV_NAME, seed=-1, render_mode="human" if RENDER else None,
                   full_episode=False)

    for episode in range(MAX_EPISODES):
        try:
            # Unique random seed & filename per episode
            episode_seed = random.randint(0, 2**31 - 1)
            filename = os.path.join(DIR_NAME, f"{episode_seed}.npz")

            # Storage for this episode
            recording_obs = []
            recording_actions = []

            # Seed Python, NumPy, and env for reproducibility
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            env.reset(seed=episode_seed)

            # Reset environment
            obs, info = env.reset()
            # obs shape: (84, 84, 1), dtype uint8

            for t in range(MAX_FRAMES):
                if RENDER:
                    env.render()

                # Store observation
                recording_obs.append(obs)

                # Random policy for now (you'll replace this later with your model)
                action = env.action_space.sample()
                recording_actions.append(action)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            episode_len = len(recording_obs)
            total_frames += episode_len

            print(
                f"Episode {episode+1}/{MAX_EPISODES} "
                f"finished after {episode_len} frames, "
                f"total recorded frames so far: {total_frames}"
            )

            # Convert to arrays and save
            recording_obs = np.array(recording_obs, dtype=np.uint8)      # (T,84,84,1)
            recording_actions = np.array(recording_actions, dtype=np.uint8)  # (T,)

            np.savez_compressed(
                filename,
                obs=recording_obs,
                action=recording_actions,
            )

        except gym.error.Error as e:
            print(f"Gym error encountered: {e}. Resetting env and continuing.")
            env.close()
            env = make_env(
                ENV_NAME,
                seed=-1,
                render_mode="human" if RENDER else None,
                full_episode=False,
            )
            continue

    env.close()
    print(f"Finished generating {MAX_EPISODES} episodes, total frames: {total_frames}")


if __name__ == "__main__":
    main()
