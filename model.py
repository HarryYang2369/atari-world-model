import numpy as np
import os 
import random
import json
import sys
import time

import torch

from env import make_env              
from ConvVAE import ConvVAE           
from rnn import (                      
    MDNRNN,
    default_hps,
    rnn_output,
    rnn_output_size,
    MODE_ZCH,
    MODE_ZC,
    MODE_Z,
    MODE_Z_HIDDEN,
    MODE_ZH,
)

EXP_MODE = MODE_ZH

# paths to pretrained models
VAE_MODEL_DIR = "models_vae_breakout"
RNN_MODEL_DIR = "model_rnn_breakout"
VAE_JSON = f"{VAE_MODEL_DIR}/vae.json"
VAE_CKPT = f"{VAE_MODEL_DIR}/vae.pt"       
RNN_CKPT = f"{RNN_MODEL_DIR}/mdn_rnn.pt"


# action space size for Breakout (adjust if different)
ACTION_SIZE = 4


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def sample(p):
    # categorical sample from probability vector p
    return np.argmax(np.random.multinomial(1, p))


def make_model(load_model=True, device=None):
    """Factory for the controller+world-model."""
    model = Model(load_model=load_model, device=device)
    return model


class Model:
    """
    Controller + world model wrapper for Breakout.

    - VAE encodes frames -> z
    - MDNRNN keeps hidden state (h,c)
    - Simple linear controller maps [z, h/c] -> action logits
    """
    
    def __init__(self, load_model=True, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.env_name = "ALE/Breakout-v5"
        
        self.vae = ConvVAE(
            z_size=128,
            batch_size=1,
            learning_rate=1e-4,
            kl_tolerance=0.5,
            is_training=False,
            reuse=False,
            gpu_mode=(self.device.type == "cuda"),
        )
        
        self.vae.load_json(VAE_JSON)
        if load_model and VAE_CKPT and os.path.exists(VAE_CKPT):
            # optional, if you saved a .pt checkpoint
            self.vae.load_checkpoint(VAE_CKPT)
        
        self.hps = default_hps()
        self.rnn = MDNRNN(self.hps, device=self.device)
        
        if load_model and RNN_CKPT:
            state_dict = torch.load(RNN_CKPT, map_location=self.device)
            self.rnn.load_state_dict(state_dict)
        self.rnn.eval()
    
        self.rnn_state = None  # (h, c) or None
        
        self.z_size = self.hps.z_size           # 128
        self.action_size = ACTION_SIZE          # 4 for Breakout
        
        self.input_size = rnn_output_size(
            z_size=self.z_size,
            rnn_size=self.hps.rnn_size,
            mode=EXP_MODE,
        )
        
        self.weight = np.random.randn(self.input_size, self.action_size)
        self.bias = np.random.randn(self.action_size)
        self.param_count = self.input_size * self.action_size + self.action_size
        
        self.render_mode = False
        
        self.last_entropy = 0.0

    def make_env(self, seed=-1, render_mode=False, full_episode=False):
        self.render_mode = render_mode
        # your make_env should know how to build Breakout env
        self.env = make_env(
            self.env_name,
            seed=seed,
            render_mode=render_mode,
            full_episode=full_episode,
        )
        
    def reset(self):
        self.rnn_state = None
        
    def encode_obs(self, obs):
        """
        Convert raw obs (84x84x1 uint8 or similar) → z (latent).

        Returns:
          z: [z_size] numpy float32
        """
        # scale to [0,1]
        result = np.copy(obs).astype(np.float32) / 255.0

        if result.ndim == 2:
            # (84,84) → (84,84,1)
            result = result[:, :, None]
        # VAE expects (1,84,84,1)
        result = result.reshape(1, result.shape[0], result.shape[1], result.shape[2])

        mu, logvar = self.vae.encode_mu_logvar(result)
        mu = mu[0]  # [z_size]

        # often using just mu (no sampling) is fine for control
        z = mu.astype(np.float32)
        return z
    
    def get_action(self, z, stochastic=True):
        """
        z: [z_size] numpy
        Returns: action_idx (int in [0, action_size-1])
        """

        #Build feature vector from z and current RNN state
        z_torch = torch.from_numpy(z.astype(np.float32)).to(self.device).unsqueeze(0)  # [1,z]
        
        if self.rnn_state is None:
            # start with zeros
            h_prev = torch.zeros(1, 1, self.hps.rnn_size, device=self.device)
            c_prev = torch.zeros(1, 1, self.hps.rnn_size, device=self.device)
        else:
            h_prev, c_prev = self.rnn_state

        feat_torch = rnn_output(z_torch, h_prev, c_prev, EXP_MODE)  # [1,input_size]
        feat = feat_torch.detach().cpu().numpy()[0]  
        
        logits = np.dot(feat, self.weight) + self.bias             # [action_size]
        probs = softmax(logits)
        
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.last_entropy = float(entropy)

        if stochastic:
            action_idx = sample(probs)
        else:
            action_idx = int(np.argmax(probs))

        a_onehot = np.zeros(self.action_size, dtype=np.float32)
        a_onehot[action_idx] = 1.0

        a_torch = torch.from_numpy(a_onehot).to(self.device).unsqueeze(0)  # [1,A]
        
        _, _, (h_next, c_next) = self.rnn.step(
            z_torch,
            a_torch,
            h_c_prev=(h_prev, c_prev),
            temperature=1.0,
        )
        
        self.rnn_state = (h_next, c_next)
        
        return action_idx
    
    
    def set_model_params(self, model_params):
        """
        model_params: flat vector of length param_count
        """
        params = np.array(model_params)
        assert params.size == self.param_count, (
            f"Expected {self.param_count} params, got {params.size}"
        )

        self.bias = params[:self.action_size]
        self.weight = params[self.action_size:].reshape(
            self.input_size, self.action_size
        )
        
    def load_model(self, filename):
        """
        Load controller params from a JSON file:
          [ [flat_param_list], ... ]
        """
        with open(filename) as f:
            data = json.load(f)
        print("loading file %s" % (filename))
        self.data = data
        model_params = np.array(data[0])
        self.set_model_params(model_params)
        
    def get_random_model_params(self, stdev=0.1):
        # controller params only (VAE/RNN are frozen)
        return np.random.standard_cauchy(self.param_count) * stdev

    def init_random_model_params(self, stdev=0.1):
        params = self.get_random_model_params(stdev=stdev)
        self.set_model_params(params)
        
def simulate(model, train_mode=False, render_mode=True,
             num_episode=5, seed=-1, max_len=-1):

    reward_list = []
    t_list = []

    max_episode_length = 10000  # safety cap, can be large for Breakout
    recording_mode = False      # set True to save trajectories

    # 🔹 reward shaping hyperparameters
    alive_bonus = 0.01          # bonus per time step survived
    entropy_coef = 0.001        # bonus for higher action entropy

    if train_mode and max_len > 0:
        max_episode_length = max_len

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    for episode in range(num_episode):
        # reset RNN state etc.
        model.reset()

        reset_out = model.env.reset()
        if isinstance(reset_out, tuple):
            # Gym >= 0.26: (obs, info)
            obs, _ = reset_out
        else:
            # older Gym: obs only
            obs = reset_out

        total_reward = 0.0
        steps = 0  # track number of steps in this episode

        #for entropy shaping
        entropy_sum = 0.0

        # recording buffer (optional)
        random_generated_int = np.random.randint(2**31 - 1)
        filename = f"record_controller/{random_generated_int}.npz"

        recording_z = []
        recording_action = []
        recording_reward = [0.0]

        for t in range(max_episode_length):
            if render_mode:
                # new-style API: render_mode was set when env was created
                model.env.render()

            # encode observation to latent z
            z = model.encode_obs(obs)            # [z_size]
            action = model.get_action(z)         # discrete int

            # record inputs & actions
            recording_z.append(z)
            recording_action.append(action)

            # step the environment
            step_out = model.env.step(action)

            if len(step_out) == 5:
                # Gym >= 0.26 / Gymnasium: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                # older Gym: (obs, reward, done, info)
                obs, reward, done, info = step_out

            recording_reward.append(float(reward))

            # base environment reward
            total_reward += reward

            #alive bonus: encourage staying alive
            total_reward += alive_bonus

            # accumulate entropy from this action
            entropy_sum += getattr(model, "last_entropy", 0.0)

            steps += 1

            if done:
                break

        # entropy bonus at end of episode (average over steps)
        if steps > 0:
            avg_entropy = entropy_sum / steps
            total_reward += entropy_coef * avg_entropy

        # last frame encode (optional, as in your original code)
        z = model.encode_obs(obs)
        action = model.get_action(z)
        recording_z.append(z)
        recording_action.append(action)

        # convert recordings to arrays
        recording_z = np.array(recording_z, dtype=np.float32)
        recording_action = np.array(recording_action, dtype=np.uint8)
        recording_reward = np.array(recording_reward, dtype=np.float32)

        # optionally save trajectories when not rendering
        if not render_mode and recording_mode:
            np.savez_compressed(
                filename,
                z=recording_z,
                action=recording_action,
                reward=recording_reward,
            )

        if render_mode:
            print(
                "total reward", total_reward,
                "timesteps", steps,
                "avg_entropy", (entropy_sum / steps) if steps > 0 else 0.0,
            )

        reward_list.append(total_reward)
        t_list.append(steps)

    return reward_list, t_list



def main():
    print("MAIN STARTED", sys.argv)
    assert len(sys.argv) > 1, "python model.py render/norender [path_to_model.json] [seed]"

    render_mode_string = str(sys.argv[1])
    if render_mode_string == "render":
        render_mode = True
    else:
        render_mode = False

    use_model = False
    if len(sys.argv) > 2:
        use_model = True
        filename = sys.argv[2]
        print("filename", filename)

    the_seed = np.random.randint(10000)
    if len(sys.argv) > 3:
        the_seed = int(sys.argv[3])
        print("seed", the_seed)
        
    # Build model + env
    model = make_model(load_model=True)
    print("controller param size", model.param_count)
    model.make_env(render_mode=render_mode)

    if use_model:
        model.load_model(filename)
    else:
        # random controller init (for quick smoke tests)
        model.init_random_model_params(stdev=np.random.rand() * 0.01)

    N_episode = 100
    if render_mode:
        N_episode = 1

    reward_list = []
    for i in range(N_episode):
        reward, steps_taken = simulate(
            model,
            train_mode=False,
            render_mode=render_mode,
            num_episode=1,
            seed=the_seed + i,
        )
        if render_mode:
            print("terminal reward", reward, "average steps taken", np.mean(steps_taken) + 1)
        else:
            print(reward[0])
        reward_list.append(reward[0])

    if not render_mode:
        print(
            "seed", the_seed,
            "average_reward", np.mean(reward_list),
            "stdev", np.std(reward_list),
        )



if __name__ == "__main__":
    main()