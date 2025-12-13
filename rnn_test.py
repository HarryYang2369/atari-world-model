import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from ConvVAE import ConvVAE
from rnn import MDNRNN, default_hps

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
EP_FILE = "record_breakout/196889587.npz"
VAE_DIR = "models_vae_breakout"
RNN_CKPT = "pt_rnn_breakout/mdnrnn_breakout.pt"

# -------------------------------------------------------------------
# Load VAE
# -------------------------------------------------------------------
vae = ConvVAE(
    z_size=128,
    batch_size=1,
    learning_rate=1e-4,
    kl_tolerance=0.5,
    is_training=False,
    gpu_mode=False,   # safer default; set True if you know TF has GPU
)

vae.load_checkpoint(os.path.join(VAE_DIR, "vae.pt"))
print("[INFO] Loaded VAE checkpoint.")

# -------------------------------------------------------------------
# Load RNN (PyTorch MDNRNN) with correct hps
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(RNN_CKPT, map_location=device)

if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "hps" in ckpt):
    raise ValueError("Checkpoint must be a dict with 'state_dict' and 'hps' as saved in rnn_train.py")

# Reconstruct hps from checkpoint
hps = default_hps()
for k, v in ckpt["hps"].items():
    setattr(hps, k, v)

print("[INFO] Loaded hps from checkpoint:", hps.__dict__)

rnn = MDNRNN(hps, device=device)
rnn.load_state_dict(ckpt["state_dict"])
rnn.eval()
print("[INFO] Loaded MDNRNN checkpoint.")

# -------------------------------------------------------------------
# Load a single recorded episode
# -------------------------------------------------------------------
ep = np.load(EP_FILE)
obs = ep["obs"].astype(np.float32) / 255.0   # [T,84,84,1]
actions = ep["action"]                       # [T]

T_roll = min(200, len(obs) - 1)             # we will roll for at most 200 steps
print(f"[INFO] Episode length: {len(obs)}, using T_roll={T_roll}")

# -------------------------------------------------------------------
# Encode all frames to z (using the same encoder as training)
# -------------------------------------------------------------------
z_all = []
for t in range(T_roll + 1):   # need z_0..z_T_roll
    frame = obs[t:t+1]        # [1,H,W,C]
    z = vae.encode(frame)     # [1,z_size] (sampled from N(mu, σ²))
    z_all.append(z[0])
z_all = np.stack(z_all, axis=0)  # [T_roll+1, z_size]

# -------------------------------------------------------------------
# Roll RNN forward (dreaming)
# -------------------------------------------------------------------
h_c = None
dream_frames = []

for t in range(T_roll):
    z_t = z_all[t:t+1]   # [1,z_size]
    a_t = int(actions[t])

    if a_t < 0 or a_t >= hps.action_size:
        raise ValueError(f"Action {a_t} at t={t} is out of range for action_size={hps.action_size}")

    a_onehot = np.eye(hps.action_size, dtype=np.float32)[[a_t]]  # [1, A]

    z_t_torch = torch.from_numpy(z_t).to(device)         # [1, z_size]
    a_t_torch = torch.from_numpy(a_onehot).to(device)    # [1, A]

    with torch.no_grad():
        # Requires MDNRNN.step(z_t, a_t, h_c, temperature)
        z_next_pred, h_c = rnn.step(z_t_torch, a_t_torch, h_c, temperature=1.0)

    # decode predicted z to image
    z_next_np = z_next_pred.cpu().numpy()   # [1, z_size]
    recon = vae.decode(z_next_np)           # [1,H,W,C] in [0,1]
    dream_frames.append(recon[0])

dream_frames = np.stack(dream_frames, axis=0)  # [T_roll,H,W,C]

# -------------------------------------------------------------------
# Visualize some timesteps
# -------------------------------------------------------------------
T_obs = obs.shape[0]
T_dream = dream_frames.shape[0]

timesteps = [0, 10, 30, 31, 32, 33, 34, 35, 60, 100, 150]
timesteps = [t for t in timesteps if (t + 1 < T_obs) and (t < T_dream)]

for t in timesteps:
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))

    # real
    axes[0].imshow(obs[t+1, :, :, 0], cmap="gray")
    axes[0].set_title(f"{t}: Real {t+1}")
    axes[0].axis("off")

    # dream
    axes[1].imshow(dream_frames[t, :, :, 0], cmap="gray")
    axes[1].set_title(f"{t}: Dream {t+1}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
