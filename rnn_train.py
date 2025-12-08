"""
Train MDN-RNN for Atari Breakout from pre-processed latent series.

Expects:
    series/series_breakout_rnn.npz with
      - mu:        [N] object, each (Ti, z_size)
      - logvar:    [N] object, each (Ti, z_size)
      - action:    [N] object, each (Ti,)
      - life_loss: [N] object, each (Ti,)  (0/1 per step: life lost at t+1)

We:
  1) sample z from N(mu, exp(logvar))
  2) train MDNRNN to predict z_{t+1} and life_loss_{t+1}
  3) save initial μ/logvar for 1000 episodes
  4) save PyTorch checkpoint of the trained RNN
"""

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim

from rnn import MDNRNN, default_hps

DATA_DIR = "series"
SERIES_FILE = "series_breakout_rnn.npz"
MODEL_SAVE_DIR = "model_rnn_breakout"
INITIAL_Z_DIR = "initial_z_breakout"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(INITIAL_Z_DIR, exist_ok=True)


def load_series(path):
    fpath = os.path.join(DATA_DIR, path)
    data = np.load(fpath, allow_pickle=True)

    mu = data["mu"]
    logvar = data["logvar"]
    action = data["action"]
    life_loss = data["life_loss"]  
    return mu, logvar, action, life_loss


def sample_z(mu, logvar):
    eps = np.random.randn(*logvar.shape).astype(np.float32)
    std = np.exp(0.5 * logvar).astype(np.float32)
    z = mu + std * eps
    return z.astype(np.float32)


def random_batch(mu, logvar, action, life_loss,
                 lengths,
                 batch_size, seq_len, z_size, action_size):
    """
    Returns:
      z_t:           [B, L, z]
      a_onehot:      [B, L, action_size]
      z_tp1:         [B, L, z]
      life_loss_tp1: [B, L]
      mask:          [B, L]
    """
    N, T = mu.shape[0], mu.shape[1]
    seq_len = min(seq_len, T)

    idx = np.random.permutation(N)[:batch_size]

    mu_b      = mu[idx, :seq_len, :]
    logvar_b  = logvar[idx, :seq_len, :]
    action_b  = action[idx, :seq_len]
    life_b    = life_loss[idx, :seq_len]
    lens_b    = lengths[idx]

    z_b = sample_z(mu_b, logvar_b)

    z_t   = z_b[:, :-1, :]
    z_tp1 = z_b[:, 1:, :]
    a_t   = action_b[:, :-1]
    life_loss_tp1 = life_b[:, 1:]

    eye = np.eye(action_size, dtype=np.float32)
    a_onehot = eye[a_t]

    life_loss_tp1 = life_loss_tp1.astype(np.float32)

    L = z_t.shape[1]
    time_idx = np.arange(L)[None, :]
    mask = (time_idx < (lens_b - 1)[:, None]).astype(np.float32)

    return z_t, a_onehot, z_tp1, life_loss_tp1, mask


def save_initial_z(mu, logvar, z_size, out_dir, num_episodes=1000):
    num = min(num_episodes, mu.shape[0])

    mu0 = mu[:num, 0, :]
    logvar0 = logvar[:num, 0, :]

    initial_mu = np.round(mu0 * 10000).astype(np.int32).tolist()
    initial_logvar = np.round(logvar0 * 10000).astype(np.int32).tolist()

    out_path = os.path.join(out_dir, "initial_z.json")
    with open(out_path, "w") as f:
        json.dump([initial_mu, initial_logvar], f, sort_keys=True, indent=0, separators=(",", ": "))
    print(f"Saved initial μ/logvar for {num} episodes to {out_path}")


def main():
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    hps = default_hps()
    print("HyperParams:", hps)

    mu_list, logvar_list, action_list, life_list = load_series(SERIES_FILE)

    print("mu container:", type(mu_list), "shape:", mu_list.shape, "dtype:", mu_list.dtype)
    print("first episode shapes:",
          mu_list[0].shape, logvar_list[0].shape,
          action_list[0].shape, life_list[0].shape)

    N_data = len(mu_list)
    z_size = mu_list[0].shape[1]

    T = min(hps.max_seq_len,
            max(m.shape[0] for m in mu_list))

    print(f"Padding/truncating episodes to T={T}, z_size={z_size}")

    mu = np.zeros((N_data, T, z_size), dtype=np.float32)
    logvar = np.zeros((N_data, T, z_size), dtype=np.float32)
    action = np.zeros((N_data, T), dtype=np.int64)
    life_loss = np.zeros((N_data, T), dtype=np.float32)
    lengths = np.zeros((N_data,), dtype=np.int32)

    for i in range(N_data):
        Ti = min(mu_list[i].shape[0], T)
        lengths[i] = Ti
        mu[i, :Ti, :]       = mu_list[i][:Ti]
        logvar[i, :Ti, :]   = logvar_list[i][:Ti]
        action[i, :Ti]      = action_list[i][:Ti]
        life_loss[i, :Ti]   = life_list[i][:Ti]

    print("lengths min/max:", lengths.min(), lengths.max())
    print(f"Packed arrays: mu {mu.shape}, logvar {logvar.shape}, action {action.shape}, life_loss {life_loss.shape}")

    batch_size = hps.batch_size
    seq_len = hps.max_seq_len + 1
    action_size = hps.action_size

    save_initial_z(mu, logvar, z_size, INITIAL_Z_DIR, num_episodes=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    rnn = MDNRNN(hps, device=device)
    optimizer = optim.Adam(rnn.parameters(), lr=hps.learning_rate)

    global_step = 0
    start_time = time.time()

    for local_step in range(hps.num_steps):
        lr = (hps.learning_rate - hps.min_learning_rate) * (hps.decay_rate ** global_step) + hps.min_learning_rate
        for g in optimizer.param_groups:
            g["lr"] = lr

        z_t, a_onehot, z_tp1, life_tp1, mask = random_batch(
            mu, logvar, action, life_loss, lengths,
            batch_size=batch_size,
            seq_len=seq_len,
            z_size=z_size,
            action_size=action_size,
        )

        z_t_torch   = torch.from_numpy(z_t)
        a_t_torch   = torch.from_numpy(a_onehot)
        z_tp1_torch = torch.from_numpy(z_tp1)
        life_torch  = torch.from_numpy(life_tp1)
        mask_torch  = torch.from_numpy(mask)

        optimizer.zero_grad()
        total_loss, mdn_l, life_l = rnn.sequence_loss(
            z_t_torch, a_t_torch, z_tp1_torch, life_torch, mask_torch
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), hps.grad_clip)
        optimizer.step()
        global_step += 1

        if global_step % 20 == 0:
            now = time.time()
            dt = now - start_time
            start_time = now
            print(
                f"step: {global_step:5d}, "
                f"lr: {lr:.6f}, "
                f"total_loss: {total_loss.item():.4f}, "
                f"mdn_loss: {mdn_l.item():.4f}, "
                f"life_loss: {life_l.item():.4f}, "
                f"dt: {dt:.3f}s"
            )

    ckpt_path = os.path.join(MODEL_SAVE_DIR, "mdn_rnn.pt")
    torch.save(rnn.state_dict(), ckpt_path)
    print(f"Saved MDN-RNN checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
