"""
Build series_breakout_rnn.npz from record_breakout using ConvVAE.

Each record file in record_breakout has:
    obs:    (T,84,84,1), uint8
    action: (T,), uint8

We encode obs using the ConvVAE to obtain:
    - mu:     [N] object, each (Ti, z_size)
    - logvar: [N] object, each (Ti, z_size)
    - action: [N] object, each (Ti,)

"""

import os
import numpy as np

from ConvVAE import ConvVAE

# ---- Paths ----
RECORD_DIR = "record_breakout"         
SERIES_DIR = "series"
SERIES_FILE = "series_breakout_rnn.npz"

VAE_MODEL_DIR = "models_vae_breakout"
VAE_JSON = os.path.join(VAE_MODEL_DIR, "vae.json")
VAE_CKPT = os.path.join(VAE_MODEL_DIR, "vae.pt")  

os.makedirs(SERIES_DIR, exist_ok=True)


def load_vae():
    """Load ConvVAE (same one used for training)."""
    vae = ConvVAE(
        z_size=128,
        batch_size=1,
        learning_rate=1e-4,
        kl_tolerance=0.5,
        is_training=False,
        reuse=False,
        gpu_mode=True,
    )

    if os.path.exists(VAE_JSON):
        vae.load_json(VAE_JSON)

    if os.path.exists(VAE_CKPT):
        vae.load_checkpoint(VAE_CKPT)

    return vae


def main():
    vae = load_vae()

    files = [f for f in os.listdir(RECORD_DIR) if f.endswith(".npz")]
    files.sort()

    print(f"Found {len(files)} episodes in '{RECORD_DIR}'")

    mu_list = []
    logvar_list = []
    action_list = []

    for i, fname in enumerate(files):
        path = os.path.join(RECORD_DIR, fname)
        data = np.load(path)

        obs = data["obs"]          # (T,84,84,1), uint8
        actions = data["action"]   # (T,), uint8
        T = obs.shape[0]

        obs_f = obs.astype(np.float32) / 255.0

        # Encode whole episode
        mu, logvar = vae.encode_mu_logvar(obs_f)

        mu_list.append(mu.astype(np.float16))
        logvar_list.append(logvar.astype(np.float16))
        action_list.append(actions.astype(np.uint8))

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"Encoded {i+1}/{len(files)} episodes")

    mu_arr = np.array(mu_list, dtype=object)
    logvar_arr = np.array(logvar_list, dtype=object)
    action_arr = np.array(action_list, dtype=object)

    out_path = os.path.join(SERIES_DIR, SERIES_FILE)
    np.savez_compressed(
        out_path,
        mu=mu_arr,
        logvar=logvar_arr,
        action=action_arr,
    )
    print(f"Saved series to {out_path}")


if __name__ == "__main__":
    main()
