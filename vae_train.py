import os
import numpy as np
from glob import glob

from ConvVAE import ConvVAE 

# ---- Hyperparams ----
z_size = 32
batch_size = 100
learning_rate = 1e-4
kl_tolerance = 0.5
NUM_EPOCH = 10
DATA_DIR = "record_breakout"
MODEL_DIR = "models_vae_breakout"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.npz")))
    all_obs = []
    for i, f in enumerate(files):
        data = np.load(f)["obs"]   # (T,84,84,1), uint8
        all_obs.append(data)
        if (i + 1) % 50 == 0:
            print(f"Loaded {i+1} files...")
    dataset = np.concatenate(all_obs, axis=0)  # (N,84,84,1)
    print("Total frames:", dataset.shape[0])
    return dataset


def get_batches(dataset, batch_size):
    N = dataset.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(0, N, batch_size):
        idx = indices[i:i+batch_size]
        batch = dataset[idx]
        # normalize to [0,1]
        batch = batch.astype(np.float32) / 255.0
        yield batch


def main():
    dataset = load_dataset(DATA_DIR)

    vae = ConvVAE(
        z_size=z_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kl_tolerance=kl_tolerance,
        is_training=True,
        gpu_mode=True,
    )

    print("train", "epoch", "iter", "loss", "recon_loss", "kl_loss")
    for epoch in range(NUM_EPOCH):
        it = 0
        for batch in get_batches(dataset, batch_size):
            loss, r_loss, kl_loss = vae.train_on_batch(batch)
            if it % 100 == 0:
                print(f"train {epoch} {it} {loss:.4f} {r_loss:.4f} {kl_loss:.4f}")
            it += 1

        # save checkpoint every epoch
        vae.save_model(MODEL_DIR)
        vae.save_json(os.path.join(MODEL_DIR, "vae.json"))

    print("Training done.")
    vae.save_model(MODEL_DIR)
    vae.save_json(os.path.join(MODEL_DIR, "vae.json"))


if __name__ == "__main__":
    main()
