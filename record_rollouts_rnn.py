import os
import random
import numpy as np
import gym
from env import make_env
import shutil

# --- Hyperparameters ---
MAX_FRAMES = 1000       # max number of steps per episode
MAX_EPISODES = 1000     # number of rollouts to generate
ENV_NAME = "ALE/Breakout-v5"
RENDER = False          # set True to see Breakout while recording

DIR_NAME = "record_breakout_rnn"
if os.path.exists(DIR_NAME):
    shutil.rmtree(DIR_NAME)
os.makedirs(DIR_NAME, exist_ok=True)


def main():
    """
    Generate rollouts from Atari Breakout for VAE + RNN training.

    Each episode is saved as a .npz file with:
        - obs:           uint8 array of shape (T, 84, 84, 1)
        - action:        uint8 array of shape (T,)
        - life_loss_tp1: uint8 array of shape (T,)  (1 if a life is lost AFTER this step)
        - lives:         uint8 array of shape (T,)  (lives after this step)
    """

    total_frames = 0

    env = make_env(
        ENV_NAME,
        seed=-1,
        render_mode="human" if RENDER else None,
        full_episode=False,          # keep normal ALE termination
    )

    for episode in range(MAX_EPISODES):
        try:
            episode_seed = random.randint(0, 2**31 - 1)
            filename = os.path.join(DIR_NAME, f"{episode_seed}.npz")

            recording_obs = []
            recording_actions = []
            recording_life_loss_tp1 = []   # CHANGED: only life-loss label
            recording_lives = []

            # seed things
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            env.reset(seed=episode_seed)

            # reset env
            obs, info = env.reset()
            # initial lives (Breakout usually 5, but we read from info if present)
            lives_prev = info.get("lives", 5)

            for t in range(MAX_FRAMES):
                if RENDER:
                    env.render()

                # store current obs
                recording_obs.append(obs)

                # random policy for now
                action = env.action_space.sample()
                recording_actions.append(action)

                # step env
                obs, reward, terminated, truncated, info = env.step(action)

                # lives after this step
                lives_next = info.get("lives", lives_prev)

                # CHANGED: life_loss_tp1 = 1 only if a life decreases
                life_loss = 1 if (lives_next < lives_prev) else 0

                recording_life_loss_tp1.append(life_loss)
                recording_lives.append(lives_next)

                lives_prev = lives_next

                # we still respect ALE env termination for recording,
                # but "game over" logic for training will be handled via lives later.
                if terminated or truncated:
                    break

            episode_len = len(recording_obs)
            total_frames += episode_len

            print(
                f"Episode {episode+1}/{MAX_EPISODES} "
                f"finished after {episode_len} frames, "
                f"total recorded frames so far: {total_frames}"
            )

            # convert to arrays and save
            recording_obs = np.array(recording_obs, dtype=np.uint8)       # (T,84,84,1)
            recording_actions = np.array(recording_actions, dtype=np.uint8)  # (T,)
            recording_life_loss_tp1 = np.array(recording_life_loss_tp1, dtype=np.uint8)  # (T,)
            recording_lives = np.array(recording_lives, dtype=np.uint8)                # (T,)

            np.savez_compressed(
                filename,
                obs=recording_obs,
                action=recording_actions,
                life_loss_tp1=recording_life_loss_tp1,   # CHANGED key name
                lives=recording_lives,
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