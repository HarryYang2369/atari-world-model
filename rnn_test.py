import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from ConvVAE import ConvVAE
from rnn import MDNRNN, default_hps

# paths
EP_FILE = "record_breakout_rnn/138923612.npz"
VAE_DIR = "models_vae_breakout"
RNN_CKPT = "model_rnn_breakout/mdn_rnn.pt"

# --- load VAE ---
vae = ConvVAE(z_size=128, batch_size=1, is_training=False, gpu_mode=True)
vae.load_checkpoint(os.path.join(VAE_DIR, "vae.pt"))  # or load_json if using quantized

# --- load RNN ---
hps = default_hps()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rnn = MDNRNN(hps, device=device)
rnn.load_state_dict(torch.load(RNN_CKPT, map_location=device))
rnn.eval()

# --- load a single recorded episode ---
ep = np.load(EP_FILE)
obs = ep["obs"].astype(np.float32) / 255.0   # [T,84,84,1] in [0,1]
actions = ep["action"]                       # [T]
T = min(200, len(obs)-1)                     # just look at first 200 steps

# encode all frames to z
z_all = []
for t in range(T+1):   # we need z_0..z_T
    frame = obs[t:t+1]  # [1,84,84,1]
    z = vae.encode(frame)  # [1,128]
    z_all.append(z[0])
z_all = np.stack(z_all, axis=0)  # [T+1,128]

# roll RNN forward
h_c = None
dream_frames = []
done_probs = []

for t in range(T):
    z_t = z_all[t:t+1]   # [1,128]
    a_t = actions[t]
    a_onehot = np.eye(hps.action_size, dtype=np.float32)[[a_t]]  # [1,4]

    z_t_torch = torch.from_numpy(z_t).to(device)
    a_t_torch = torch.from_numpy(a_onehot).to(device)

    with torch.no_grad():
        z_next_pred, done_prob, h_c = rnn.step(z_t_torch, a_t_torch, h_c, temperature=1.0)

    done_probs.append(done_prob.cpu().numpy()[0])

    # decode predicted z to image
    z_next_np = z_next_pred.cpu().numpy()
    recon = vae.decode(z_next_np)  # [1,84,84,1], in [0,1]
    dream_frames.append(recon[0])

dream_frames = np.stack(dream_frames, axis=0)  # [T,84,84,1]

# --- visualize a few timesteps ---
T = obs.shape[0]  # number of frames

timesteps = [0, 10, 30, 31, 32, 33, 34, 35, 60, 100, 150]
timesteps = [t for t in timesteps if t + 1 < T]  # filter invalid ones

for t in timesteps:
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))
    axes[0].imshow(obs[t+1, :, :, 0], cmap="gray")
    axes[0].set_title(f"{t}:Real {t+1}")
    axes[0].axis("off")

    axes[1].imshow(dream_frames[t, :, :, 0], cmap="gray")
    axes[1].set_title(f"{t}: Dream {t+1} (p_done={done_probs[t]:.2f})")
    axes[1].axis("off")

    plt.show()

