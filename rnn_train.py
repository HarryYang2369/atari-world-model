# rnn_train.py
#
# Train PyTorch MDN-RNN for Atari Breakout from pre-processed latent series.
#
# Expected NPZ file: series/series_breakout_rnn.npz with:
#   - mu:        [N] object, each (Ti, z_size)
#   - logvar:    [N] object, each (Ti, z_size)
#   - action:    [N] object, each (Ti,)  (int action ids)
#
# We:
#   1) sample z ~ N(mu, exp(logvar))
#   2) train MDNRNN (PyTorch) to predict z_{t+1} from (z_t, a_t)
#   3) save initial μ/logvar for 1000 episodes (for dreaming env)
#   4) save MDN-RNN weights to mdnrnn_breakout.pt

import os
import json
import time

import numpy as np
import torch

from rnn import MDNRNN, default_hps  # your PyTorch MDNRNN & HyperParams

import matplotlib
matplotlib.use("Agg")  # headless backend for clusters
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------
DATA_DIR = "series"
SERIES_FILE = "series_breakout_rnn.npz"

MODEL_SAVE_PATH = "pt_rnn_breakout"
INITIAL_Z_SAVE_PATH = "pt_initial_z_breakout"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(INITIAL_Z_SAVE_PATH, exist_ok=True)


# --------------------------------------------------------------------
# Load series from NPZ (object arrays)
# --------------------------------------------------------------------
def load_series_npz():
    """
    Load latent series from NPZ:

      mu:     object array of length N, each (Ti, z_size)
      logvar: object array of length N, each (Ti, z_size)
      action: object array of length N, each (Ti,)

    No life_loss is used.
    """
    path = os.path.join(DATA_DIR, SERIES_FILE)
    data = np.load(path, allow_pickle=True)

    mu_list = data["mu"]       # object array of (Ti, z_size)
    logvar_list = data["logvar"]
    action_list = data["action"]

    return mu_list, logvar_list, action_list


def get_frame_count(mu_list):
    """Total number of frames across all episodes."""
    return sum(m.shape[0] for m in mu_list)


# --------------------------------------------------------------------
# Flatten episodes and create batches
# --------------------------------------------------------------------
def create_batches_from_episodes(
    mu_list,
    logvar_list,
    action_list,
    batch_size=64,
    seq_length=200,
    z_size=128,
):
    """
    Build batches from contiguous chunks of each episode.
    No crossing episode boundaries.
    """
    # Collect all (z_mu, z_logvar, actions) chunks of length seq_length
    seq_mu = []
    seq_logvar = []
    seq_action = []

    for mu, logvar, act in zip(mu_list, logvar_list, action_list):
        T = mu.shape[0]
        # we need at least seq_length+1 frames to form (z_t, z_{t+1}) pairs inside
        if T <= seq_length:
            continue

        # slide in non-overlapping chunks of length seq_length
        num_chunks = T // seq_length
        for c in range(num_chunks):
            start = c * seq_length
            end = start + seq_length
            seq_mu.append(mu[start:end])          # (L, z_size)
            seq_logvar.append(logvar[start:end])  # (L, z_size)
            seq_action.append(act[start:end])     # (L,)

    num_seqs = len(seq_mu)
    if num_seqs == 0:
        raise ValueError("No sequences long enough for the given seq_length.")

    # Now pack these sequences into batches
    num_batches = num_seqs // batch_size
    num_seqs_adjusted = num_batches * batch_size

    seq_mu = np.array(seq_mu[:num_seqs_adjusted])       # (N, L, z)
    seq_logvar = np.array(seq_logvar[:num_seqs_adjusted])
    seq_action = np.array(seq_action[:num_seqs_adjusted])

    # reshape to [B, L, ...]
    data_mu = seq_mu.reshape(num_batches, batch_size, seq_length, z_size)
    data_logvar = seq_logvar.reshape(num_batches, batch_size, seq_length, z_size)
    data_action = seq_action.reshape(num_batches, batch_size, seq_length)

    # we want list of length num_batches with shape [B, L, ...]
    data_mu = [data_mu[i] for i in range(num_batches)]
    data_logvar = [data_logvar[i] for i in range(num_batches)]
    data_action = [data_action[i] for i in range(num_batches)]

    return data_mu, data_logvar, data_action



def get_batch(batch_idx, data_mu, data_logvar, data_action):
    """
    Returns:
      batch_z:      [B, L, z_size] sampled from (mu, logvar)
      batch_action: [B, L] int actions
    """
    batch_mu = data_mu[batch_idx]          # [B, L, z]
    batch_logvar = data_logvar[batch_idx]  # [B, L, z]
    batch_action = data_action[batch_idx]  # [B, L]

    # Sample z ~ N(mu, exp(logvar))
    batch_s = batch_logvar.shape
    eps = np.random.randn(*batch_s).astype(np.float32)
    std = np.exp(batch_logvar.astype(np.float32) / 2.0)
    batch_z = batch_mu.astype(np.float32) + std * eps

    return (
        batch_z.astype(np.float32),
        batch_action.astype(np.int32),
    )


# --------------------------------------------------------------------
# Save initial mu/logvar (for dreaming env)
# --------------------------------------------------------------------
def save_initial_z(mu_list, logvar_list, z_size, out_dir, num_episodes=1000):
    """
    Save first-step μ, logvar (scaled and quantized) for up to num_episodes.
    Used by the world-models-style dreaming environment.
    """
    num = min(num_episodes, len(mu_list))
    initial_mu = []
    initial_logvar = []
    for i in range(num):
        mu0 = np.copy(mu_list[i][0, :] * 10000).astype(np.int32).tolist()
        logvar0 = np.copy(logvar_list[i][0, :] * 10000).astype(np.int32).tolist()
        initial_mu.append(mu0)
        initial_logvar.append(logvar0)

    out_path = os.path.join(out_dir, "initial_z.json")
    with open(out_path, "wt") as outfile:
        json.dump([initial_mu, initial_logvar], outfile,
                  sort_keys=True, indent=0, separators=(",", ": "))
    print(f"[INFO] Saved initial_z for {num} episodes to {out_path}")


# --------------------------------------------------------------------
# Main training loop (PyTorch)
# --------------------------------------------------------------------
def train_mdnrnn_breakout():
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) Load data
    mu_list, logvar_list, action_list = load_series_npz()
    print("[INFO] Loaded episodes:", len(mu_list))
    print(
        "[INFO] First episode shapes:",
        mu_list[0].shape,
        logvar_list[0].shape,
        action_list[0].shape,
    )

    z_size = mu_list[0].shape[1]
    # Infer number of actions from data
    num_actions = int(max(a.max() for a in action_list) + 1)
    print(f"[INFO] Detected z_size={z_size}, action_size={num_actions}")

    # 2) Hyperparameters
    hps = default_hps()
    # Adjust to actual data
    hps.z_size = z_size
    hps.action_size = num_actions
    # You can also tweak:
    # hps.max_seq_len = 500
    # hps.batch_size  = 100

    # 3) Save initial_z.json (for dreaming env)
    save_initial_z(mu_list, logvar_list, z_size, INITIAL_Z_SAVE_PATH, num_episodes=1000)

    # 4) Build model + optimizer
    model = MDNRNN(hps, device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=hps.learning_rate)

    # Loss tracking
    steps_history = []
    loss_history = []

    global_step = 0
    start = time.time()

    NUM_EPOCHS = 600  

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[INFO] ===== Epoch {epoch} =====")
        print("[INFO] Preparing data for epoch", epoch)

        data_mu, data_logvar, data_action = create_batches_from_episodes(
            mu_list,
            logvar_list,
            action_list,
            batch_size=hps.batch_size,
            seq_length=hps.max_seq_len,
            z_size=z_size,
        )

        num_batches = len(data_mu)
        print("[INFO] Number of batches:", num_batches)
        end = time.time()
        print("[INFO] Time to create batches: %.2f s" % (end - start))
        start = time.time()

        for local_step in range(num_batches):
            batch_z, batch_action = get_batch(
                local_step, data_mu, data_logvar, data_action
            )

            # One-hot encode actions: [B, L] -> [B, L, A]
            eye = np.eye(hps.action_size, dtype=np.float32)
            batch_action_oh = eye[batch_action]  # [B, L, A]

            # Convert to torch tensors
            z_torch = torch.from_numpy(batch_z).to(device)          # [B, L, z]
            a_torch = torch.from_numpy(batch_action_oh).to(device)  # [B, L, A]

            # Learning rate schedule (same functional form as old script)
            curr_lr = (
                (hps.learning_rate - hps.min_learning_rate)
                * (hps.decay_rate ** global_step)
                + hps.min_learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            optimizer.zero_grad()
            loss = model.sequence_loss(z_torch, a_torch)  # MDN NLL on z_{t+1}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip)
            optimizer.step()

            global_step += 1

            # Log
            steps_history.append(global_step)
            loss_history.append(float(loss.item()))

            if global_step % 20 == 0:
                end = time.time()
                print(
                    "[INFO] step: %d, lr: %.6f, loss: %.4f, batch_time: %.2f s"
                    % (global_step, curr_lr, loss.item(), end - start)
                )
                start = time.time()

    # Save model state_dict & hps
    save_path = os.path.join(MODEL_SAVE_PATH, "mdnrnn_breakout.pt")
    torch.save(
        {"state_dict": model.state_dict(), "hps": hps.__dict__},
        save_path,
    )
    print("[INFO] Saved MDN-RNN weights to", save_path)

    # 6) Plot training curve
    if len(steps_history) > 0:
        plt.figure()
        plt.plot(steps_history, loss_history, label="total loss (MDN NLL)")
        plt.xlabel("Global step")
        plt.ylabel("Loss")
        plt.title("MDN-RNN training loss (PyTorch)")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(MODEL_SAVE_PATH, "training_loss.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("[INFO] Saved training loss plot to", plot_path)
    else:
        print("[WARN] No training steps recorded; skipping loss plot.")


if __name__ == "__main__":
    train_mdnrnn_breakout()
