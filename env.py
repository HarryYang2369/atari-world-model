import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from skimage.transform import resize
from gym.spaces import Box
import gym


SCREEN_X = 84
SCREEN_Y = 84


def _process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Process raw Atari frame (210x160x3 uint8) into 84x84 uint8 grayscale image.
    Steps:
      1. Crop play area (remove top scoreboard and bottom border)
      2. Resize to 84x84
      3. Convert to grayscale
      4. Return (84,84) uint8 in [0, 255]
    """

    #  Crop play area: raw frame shape is (210, 160, 3)
    frame = frame[34:34 + 160, :, :]     # (160, 160, 3), uint8

    #  Resize to 84x84 (still 3-channel)
    frame = resize(
        frame,
        (SCREEN_Y, SCREEN_X),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)                   # (84, 84, 3), uint8

    #  Convert to grayscale using standard luminance weights
    img = (
        0.299 * frame[:, :, 0] +
        0.587 * frame[:, :, 1] +
        0.114 * frame[:, :, 2]
    ).astype(np.uint8)                   # (84, 84), uint8

    return img


class AtariWorldModelWrapper(gym.Wrapper):
    """
    Wrapper that:
      - Takes raw Atari frames from ALE (210x160x3)
      - Applies our crop+resize+grayscale preprocessing
      - Exposes observations as (84,84,1) uint8 in [0,255]
    """
    def __init__(self, env, full_episode: bool = False):
        super().__init__(env)
        self.full_episode = full_episode

        # Processed obs: 84x84x1, uint8 [0,255]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(SCREEN_Y, SCREEN_X, 1),
            dtype=np.uint8,
        )

    def step(self, action):
        # gym 0.26 API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)

        frame = _process_frame(obs)      # (84,84), uint8
        frame = frame[..., None]         # (84,84,1), uint8

        if self.full_episode:
            terminated = False
            truncated = False

        return frame, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = _process_frame(obs)[..., None]   # (84,84,1), uint8
        return frame, info


def make_env(env_name: str,
             seed: int = -1,
             render_mode: str | None = None,
             full_episode: bool = False):
    """
    Factory for Atari env wrapped with our World Model preprocessing.
    Example call: env = make_env("ALE/Breakout-v5")
    """

    env = gym.make(env_name, render_mode=render_mode)
    env = AtariWorldModelWrapper(env, full_episode=full_episode)

    if seed >= 0:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


if __name__ == "__main__":
    ENV_NAME = "ALE/Breakout-v5"

    env = make_env(ENV_NAME, seed=0, render_mode="human", full_episode=False)

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)   # (84, 84, 1)
    print("Initial obs dtype:", obs.dtype)   # uint8
    print("Initial obs min/max:", obs.min(), obs.max())

    total_reward = 0.0
    for t in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if t % 50 == 0:
            print(f"t={t}, reward={reward:.3f}, total_reward={total_reward:.3f}")
            print("obs shape:", obs.shape,
                  "| dtype:", obs.dtype,
                  "| min/max:", obs.min(), obs.max())

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            total_reward = 0.0

        env.render()

    env.close()
