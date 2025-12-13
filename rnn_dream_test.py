# rnn_dream_test.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from ConvVAE import ConvVAE
from rnn import MDNRNN, default_hps  # no need to import HyperParams directly

# -------------------------------------------------------
# Config – adjust paths if needed
# -------------------------------------------------------
VAE_MODEL_DIR = "models_vae_breakout"
RNN_MODEL_DIR = "pt_rnn_breakout"

# ⚠️ Use the *same* rollout directory that you used to train VAE + RNN
ROLL_OUT_DIR  = "record_breakout"   # not record_breakout_rnn anymore

VAE_JSON = os.path.join(VAE_MODEL_DIR, "vae.json")
VAE_CKPT = os.path.join(VAE_MODEL_DIR, "vae.pt")        # TF checkpoint path
RNN_CKPT = os.path.join(RNN_MODEL_DIR, "mdnrnn_breakout.pt")

MAX_DREAM_LEN = 600


# -------------------------------------------------------
# Loading helpers
# -------------------------------------------------------
def load_vae(device):
    vae = ConvVAE(
        z_size=128,
        batch_size=1,
        learning_rate=1e-4,
        kl_tolerance=0.5,
        is_training=False,
        reuse=False,
        gpu_mode=(device.type == "cuda"),
    )

    if os.path.exists(VAE_CKPT):
        print(f"[INFO] Loading VAE checkpoint from {VAE_CKPT}")
        vae.load_checkpoint(VAE_CKPT)
    elif os.path.exists(VAE_JSON):
        print(f"[INFO] VAE .pt not found, loading from {VAE_JSON}")
        vae.load_json(VAE_JSON)
    else:
        raise FileNotFoundError(
            f"No VAE checkpoint found at {VAE_CKPT} or {VAE_JSON}"
        )

    return vae


def load_rnn(device):
    """Load trained MDN-RNN (PyTorch version, z-only)."""
    if not os.path.exists(RNN_CKPT):
        raise FileNotFoundError(f"RNN checkpoint not found at {RNN_CKPT}")

    ckpt = torch.load(RNN_CKPT, map_location=device)

    # 1) Rebuild hps from checkpoint if present, otherwise use defaults
    if isinstance(ckpt, dict) and "hps" in ckpt:
        print("[INFO] Reconstructing HyperParams from checkpoint.")
        hps = default_hps()
        for k, v in ckpt["hps"].items():
            setattr(hps, k, v)
    else:
        print("[WARN] No hps in checkpoint, falling back to default_hps().")
        hps = default_hps()

    # 2) Build model with those hps
    rnn = MDNRNN(hps, device=device)

    # 3) Extract real state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # fallback: assume checkpoint itself is a state_dict

    # 4) Load weights (allow non-critical mismatches)
    missing, unexpected = rnn.load_state_dict(state_dict, strict=False)
    print("[INFO] Loaded RNN. Missing keys:", missing)
    print("[INFO] Unexpected keys (ignored):", unexpected)

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
    print(f"[INFO] Loaded episode '{fname}' with T={obs.shape[0]}")
    return obs, actions


# -------------------------------------------------------
# VAE encode / decode
# -------------------------------------------------------
def encode_episode(vae, obs):
    """
    Encode obs sequence to z using VAE.
    Assumes ConvVAE.encode accepts [T,84,84,1] and returns [T, z_size].
    """
    obs_f = obs.astype(np.float32) / 255.0  # (T,84,84,1) in [0,1]
    z = vae.encode(obs_f)                   # (T, z_size)
    return z.astype(np.float32)


def decode_latents_to_frames(vae, z_seq):
    """Decode [L, z_size] latents to [L,84,84,1] frames in [0,1]."""
    frames = vae.decode(z_seq.astype(np.float32))
    return frames  # (L,84,84,1)


def one_hot_actions(actions, action_size):
    """Convert [T] int actions to [T, action_size] one-hot."""
    eye = np.eye(action_size, dtype=np.float32)
    return eye[actions]  # (T, action_size)


# -------------------------------------------------------
# Dream rollout: open-loop in latent space
# -------------------------------------------------------
def dream_rollout_open_loop(rnn, hps, z0, action_onehot, device,
                            max_dream_len=MAX_DREAM_LEN,
                            temperature=1.0):
    """
    Open-loop 'dream' in latent space:
      - start from real z0
      - at each step, feed previous *predicted* z and an action
      - actions are taken from the recorded sequence (clamped at the end)

    Args:
      rnn:           MDNRNN model
      hps:           HyperParams (for action_size)
      z0:            [1, z_size] initial latent
      action_onehot: [T_real, action_size] recorded one-hot actions
      device:        torch.device
      max_dream_len: number of steps to roll out
      temperature:   MDN sampling temperature

    Returns:
      z_dream:       [L, z_size] dreamed latents
    """
    z_size = z0.shape[1]
    T_real_actions = action_onehot.shape[0]

    z_dream = np.zeros((max_dream_len, z_size), dtype=np.float32)

    z_t = z0.copy()    # [1, z_size], numpy
    state = None       # (h, c) or None

    with torch.no_grad():
        for t in range(max_dream_len):
            # pick action from recorded sequence; repeat last when t > T_real
            idx = min(t, T_real_actions - 1)
            a_t = action_onehot[idx:idx+1]  # [1, A]

            z_torch = torch.from_numpy(z_t).to(device)
            a_torch = torch.from_numpy(a_t).to(device)

            # MDNRNN.step must return (z_next, (h_next, c_next))
            z_next, state = rnn.step(z_torch, a_torch, state, temperature=temperature)

            z_np = z_next.cpu().numpy()  # [1, z_size]
            z_dream[t] = z_np[0]

            # feed prediction back in (open-loop)
            z_t = z_np

    return z_dream


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) Load models
    vae = load_vae(device)
    rnn, hps = load_rnn(device)

    # 2) Load one recorded episode
    obs, actions = load_one_episode()           # obs (T,84,84,1), actions (T,)
    T_real = obs.shape[0]

    # 3) Encode to latents (real sequence)
    z_seq = encode_episode(vae, obs)           # (T, z_size)
    z_size = z_seq.shape[1]
    print("[INFO] z_seq shape:", z_seq.shape)

    # 4) One-hot actions
    action_onehot = one_hot_actions(actions, hps.action_size)  # (T, action_size)

    # 5) Dream rollout starting from z_0
    max_len = min(MAX_DREAM_LEN, T_real)
    z0 = z_seq[0:1]  # [1, z_size]
    z_dream = dream_rollout_open_loop(
        rnn,
        hps,
        z0,
        action_onehot,
        device,
        max_dream_len=max_len,
        temperature=1.0,
    )
    dream_len = z_dream.shape[0]
    print("[INFO] z_dream shape:", z_dream.shape)

    # 6) Decode dream latents to frames
    dream_frames = decode_latents_to_frames(vae, z_dream)  # (L,84,84,1)

    # 7) Visual inspection: compare real vs dream at several timesteps
    timesteps = [0, 1, 2, 3, 4, 10, 30, 60, 100, 150, 200]
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
        axes[1].set_title(f"Dream t={t}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
