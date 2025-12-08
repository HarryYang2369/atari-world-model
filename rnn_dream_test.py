# rnn_dream_test.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from ConvVAE import ConvVAE          
from rnn import MDNRNN, default_hps  


# -------------------------------------------------------
# Config – ADJUST PATHS IF NEEDED
# -------------------------------------------------------
VAE_MODEL_DIR = "models_vae_breakout"
RNN_MODEL_DIR = "model_rnn_breakout"
ROLL_OUT_DIR  = "record_breakout_rnn"   # where obs/action were recorded

VAE_JSON = os.path.join(VAE_MODEL_DIR, "vae.json")
VAE_CKPT = os.path.join(VAE_MODEL_DIR, "vae.pt")       # optional, if you also save .pt
RNN_CKPT = os.path.join(RNN_MODEL_DIR, "mdn_rnn.pt")

MAX_DREAM_LEN = 600        # maximum length of dream rollout
INIT_LIVES = 5             # starting lives in Breakout


# -------------------------------------------------------
# Loading helpers
# -------------------------------------------------------
def load_vae(device):
    """Load trained ConvVAE (latent dim 128)."""
    vae = ConvVAE(
        z_size=128,
        batch_size=1,
        learning_rate=1e-4,
        kl_tolerance=0.5,
        is_training=False,
        reuse=False,
        gpu_mode=(device.type == "cuda"),
    )
    # Quantized params (World Models style)
    vae.load_json(VAE_JSON)

    # (Optional) also load full PyTorch checkpoint if you saved it:
    if os.path.exists(VAE_CKPT):
        vae.load_checkpoint(VAE_CKPT)

    return vae


def load_rnn(device):
    """Load trained MDN-RNN (single 'life loss' head)."""
    hps = default_hps()
    rnn = MDNRNN(hps, device=device)
    state_dict = torch.load(RNN_CKPT, map_location=device)
    rnn.load_state_dict(state_dict)
    rnn.eval()
    return rnn, hps


def load_one_episode():
    """Pick a random .npz rollout and return obs/actions."""
    files = [f for f in os.listdir(ROLL_OUT_DIR) if f.endswith(".npz")]
    assert files, f"No .npz files found in {ROLL_OUT_DIR}"

    fname = random.choice(files)
    path = os.path.join(ROLL_OUT_DIR, fname)
    data = np.load(path)

    obs = data["obs"]        # (T,84,84,1), uint8
    actions = data["action"] # (T,), uint8
    print(f"Loaded episode '{fname}' with T={obs.shape[0]}")
    return obs, actions


# -------------------------------------------------------
# VAE encode / decode
# -------------------------------------------------------
def encode_episode(vae, obs, device):
    """Encode obs sequence to z using VAE μ (no sampling)."""
    # scale to [0,1]
    obs_f = obs.astype(np.float32) / 255.0  # (T,84,84,1)
    mu, _ = vae.encode_mu_logvar(obs_f)
    # mu: (T, z_size)
    return mu.astype(np.float32)


def decode_latents_to_frames(vae, z_seq):
    """Decode [L, z_size] latents to [L,84,84,1] frames in [0,1]."""
    frames = vae.decode(z_seq.astype(np.float32))
    # frames: (L,84,84,1), float32 [0,1]
    return frames


def one_hot_actions(actions, action_size):
    """Convert [T] int actions to [T, action_size] one-hot."""
    eye = np.eye(action_size, dtype=np.float32)
    return eye[actions]  # (T, action_size)


# -------------------------------------------------------
# Dream rollout with external life counter & game over
# -------------------------------------------------------
def long_dream_rollout(rnn, z_seq, action_onehot, device,
                       max_dream_len=MAX_DREAM_LEN,
                       init_lives=INIT_LIVES):
    """
    Generate a 'dream' in latent space using MDN-RNN.

    The RNN predicts p_life_loss at each step, and we keep an external
    life counter. When lives reach 0, we mark game over and stop.

    Args:
      rnn:             MDNRNN model
      z_seq:           [T_real, z_size] real latents (for initial z_0)
      action_onehot:   [T_real, A] real one-hot actions
      device:          torch.device
      max_dream_len:   maximum number of dream steps
      init_lives:      starting lives

    Returns:
      z_dream:         [L, z_size] dream latents (L ≤ max_dream_len)
      p_life_loss:     [L] predicted life-loss probabilities
      lives_hist:      [L] lives after each dream step
      game_over_step:  int or None (index where lives hit 0)
    """
    z_size = z_seq.shape[1]
    T_real = z_seq.shape[0]

    # storage (upper bound max_dream_len, we'll slice to actual length later)
    z_dream = np.zeros((max_dream_len, z_size), dtype=np.float32)
    p_life_loss = np.zeros((max_dream_len,), dtype=np.float32)
    lives_hist = np.zeros((max_dream_len,), dtype=np.int32)

    # start at first latent and initial lives
    z_t = z_seq[0:1]      # [1, z_size]
    state = None
    lives = init_lives
    game_over_step = None

    with torch.no_grad():
        for t in range(max_dream_len):
            # choose action: use real sequence while available, then repeat last
            idx = min(t, action_onehot.shape[0] - 1)
            a_t = action_onehot[idx:idx+1]  # [1, A]

            z_torch = torch.from_numpy(z_t).to(device)
            a_torch = torch.from_numpy(a_t).to(device)

            # step RNN: second output is p_life_loss
            z_next, p_life, state = rnn.step(
                z_torch, a_torch, state, temperature=1.0
            )

            z_np = z_next.cpu().numpy()          # [1, z_size]
            p_np = p_life.cpu().numpy()[0]       # scalar

            z_dream[t] = z_np[0]
            p_life_loss[t] = p_np

            # sample whether a life is lost at this step
            life_lost = (np.random.rand() < p_np)
            if life_lost:
                lives -= 1
            lives_hist[t] = lives

            # update for next step
            z_t = z_np

            # check game over
            if lives <= 0:
                game_over_step = t
                print(f"[dream] Game over at step {t}, lives reached 0")
                # stop dream here
                actual_len = t + 1
                break
        else:
            # if we never broke, we used full max_dream_len
            actual_len = max_dream_len

    # slice to actual length
    z_dream = z_dream[:actual_len]
    p_life_loss = p_life_loss[:actual_len]
    lives_hist = lives_hist[:actual_len]

    return z_dream, p_life_loss, lives_hist, game_over_step


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load models
    vae = load_vae(device)
    rnn, hps = load_rnn(device)

    # 2) Load one recorded episode
    obs, actions = load_one_episode()           # obs (T,84,84,1), actions (T,)
    T_real = obs.shape[0]

    # 3) Encode to latents
    z_seq = encode_episode(vae, obs, device)    # (T, z_size)
    z_size = z_seq.shape[1]
    print("z_seq shape:", z_seq.shape)

    # 4) One-hot actions
    action_onehot = one_hot_actions(actions, hps.action_size)  # (T,4)

    # 5) Long dream rollout in latent space with life counter
    max_len = min(MAX_DREAM_LEN, T_real * 2)   # e.g. up to 2× real length
    z_dream, p_life_loss, lives_hist, game_over_step = long_dream_rollout(
        rnn, z_seq, action_onehot, device,
        max_dream_len=max_len,
        init_lives=INIT_LIVES,
    )
    dream_len = z_dream.shape[0]
    print("z_dream shape:", z_dream.shape)
    print("p_life_loss min/max:", p_life_loss.min(), p_life_loss.max())
    print("lives_hist first/last:", lives_hist[0], lives_hist[-1])
    print("game_over_step:", game_over_step)

    # 6) Decode dream latents to frames
    dream_frames = decode_latents_to_frames(vae, z_dream)  # (L,84,84,1)

    # 7) Visual inspection: compare real vs dream at several timesteps
    timesteps = [0,1,2,3,4, 10, 30, 60, 100, 150]
    for t in timesteps:
        if t >= dream_len:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        # Real frame at t (if available)
        if t < T_real:
            axes[0].imshow(obs[t, :, :, 0], cmap="gray")
            axes[0].set_title(f"Real t={t}")
        else:
            axes[0].imshow(np.zeros_like(obs[0, :, :, 0]), cmap="gray")
            axes[0].set_title(f"Real t={t} (no data)")
        axes[0].axis("off")

        # Dream frame at t
        axes[1].imshow(dream_frames[t, :, :, 0], cmap="gray")
        axes[1].set_title(
            f"Dream t={t}\n"
            f"p_life_loss={p_life_loss[t]:.2f}, lives={lives_hist[t]}"
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    # 8) Plot life-loss probability and lives over time
    fig, ax1 = plt.subplots(figsize=(7, 3))
    ax1.plot(p_life_loss, label="p_life_loss")
    ax1.set_xlabel("t")
    ax1.set_ylabel("p_life_loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lives_hist, label="lives", linestyle="--")
    ax2.set_ylabel("lives")

    if game_over_step is not None:
        ax1.axvline(game_over_step, color="red", linestyle=":", alpha=0.7,
                    label="game over")

    fig.legend(loc="upper right")
    plt.title("Life-loss probability & lives during dream rollout")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
