import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Modes 
# ---------------------------------------------------------------------
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4


# ---------------------------------------------------------------------
# Hyperparameters (replacing TF's namedtuple HyperParams)
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    num_steps: int = 3000
    max_seq_len: int = 1000
    input_seq_width: int = 128 + 4
    output_seq_width: int = 128
    rnn_size: int = 512
    batch_size: int = 100
    grad_clip: float = 1.0
    num_mixture: int = 5
    learning_rate: float = 1e-3
    decay_rate: float = 1.0
    min_learning_rate: float = 1e-5
    use_layer_norm: int = 0
    use_recurrent_dropout: int = 0
    recurrent_dropout_prob: float = 0.9
    use_input_dropout: int = 0
    input_dropout_prob: float = 0.9
    use_output_dropout: int = 0
    output_dropout_prob: float = 0.9
    is_training: int = 1
    z_size: int = 128
    action_size: int = 4
    done_loss_scale: float = 5.0   # single head scaling factor



def default_hps() -> HyperParams:
    """Equivalent to the TF default_hps(), but with Atari-specific sizes."""
    return HyperParams()


# ---------------------------------------------------------------------
# MDN-RNN in PyTorch (rewriting TF MDNRNN class)
# ---------------------------------------------------------------------
class MDNRNN(nn.Module):
    """
    MDN-RNN for Atari Breakout with:
      - MDN prediction for z_{t+1}
      - single life-loss probability head p(life_lost at t+1)

    total_loss = mdn_loss + done_loss_scale * life_loss_loss
    """

    def __init__(self, hps, device: torch.device | None = None):
        super().__init__()
        self.hps = hps
        self.z_size = hps.z_size
        self.action_size = hps.action_size
        self.num_mixture = hps.num_mixture
        self.rnn_size = hps.rnn_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        input_size = self.z_size + self.action_size   # [z_t, a_t]
        output_z = self.z_size                        # predict z_{t+1}

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.rnn_size,
            num_layers=1,
            batch_first=True,
        )

        # MDN head: for each latent dim we output K mixtures: (logmix, mean, logstd)
        self.mdn_linear = nn.Linear(
            self.rnn_size,
            output_z * self.num_mixture * 3,
        )

        # SINGLE head for life-loss probability
        self.done_linear = nn.Linear(self.rnn_size, 1)

        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, z_t, a_t, h0=None, c0=None):
        """
        z_t: [B, T, z_size]
        a_t: [B, T, action_size]

        Returns:
          logmix:      [B, T, z_size, K]
          mean:        [B, T, z_size, K]
          logstd:      [B, T, z_size, K]
          done_logits: [B, T]    (logits for p(life_lost at t+1))
          (h_n, c_n):  final LSTM state
        """
        B, T, _ = z_t.shape

        x = torch.cat([z_t, a_t], dim=-1)  # [B,T,z+A]

        if h0 is None or c0 is None:
            h0 = torch.zeros(1, B, self.rnn_size, device=self.device)
            c0 = torch.zeros(1, B, self.rnn_size, device=self.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))  # [B,T,H]

        # MDN head
        mdn_out = self.mdn_linear(lstm_out)  # [B,T,z*K*3]
        mdn_out = mdn_out.view(B, T, self.z_size, self.num_mixture * 3)
        
        logmix, mean, logstd = torch.split(mdn_out, self.num_mixture, dim=-1)
        logmix = F.log_softmax(logmix, dim=-1)  # mixture weights

        # life-loss logits
        done_logits = self.done_linear(lstm_out).squeeze(-1)  # [B,T]

        return logmix, mean, logstd, done_logits, (h_n, c_n)

    # ------------------------------------------------------------------
    # MDN loss (per timestep)
    # ------------------------------------------------------------------
    def mdn_loss_per_timestep(self, z_target, logmix, mean, logstd):
        """
        z_target: [B,T,z]
        logmix, mean, logstd: [B,T,z,K]

        Returns:
          nll_per_timestep: [B,T]
        """
        y = z_target.unsqueeze(-1)  # [B,T,z,1]

        log_two_pi = math.log(2.0 * math.pi)
        inv_std = torch.exp(-logstd)
        normed = (y - mean) * inv_std
        log_prob = -0.5 * (normed ** 2 + 2.0 * logstd + log_two_pi)  # [B,T,z,K]

        log_prob = logmix + log_prob
        log_prob = torch.logsumexp(log_prob, dim=-1)  # [B,T,z]

        nll = -log_prob.mean(dim=-1)  # average over z dim → [B,T]
        return nll

    # ------------------------------------------------------------------
    # Sequence loss with optional masking
    # ------------------------------------------------------------------
    def sequence_loss(self, z_t, a_t, z_tp1, life_loss_tp1, mask=None):
        """
        z_t:           [B, T, z_size]   latent at time t
        a_t:           [B, T, action_size] one-hot actions at time t
        z_tp1:         [B, T, z_size]   latent target at time t+1
        life_loss_tp1: [B, T]           0/1 flag: 1 if life is lost at t+1
        mask:          [B, T]           1 = valid timestep, 0 = padding (optional)

        Returns:
          total_loss, mdn_loss, life_loss
        """
        z_t          = z_t.to(self.device)
        a_t          = a_t.to(self.device)
        z_tp1        = z_tp1.to(self.device)
        life_loss_tp1 = life_loss_tp1.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        logmix, mean, logstd, done_logits, _ = self.forward(z_t, a_t)

        # MDN per-timestep NLL [B,T]
        mdn_l_per = self.mdn_loss_per_timestep(z_tp1, logmix, mean, logstd)

        # life-loss BCE per timestep [B,T]
        life_l_per = F.binary_cross_entropy_with_logits(
            done_logits, life_loss_tp1, reduction="none"
        )

        if mask is not None:
            # aggregate only on valid timesteps
            mdn_l = (mdn_l_per * mask).sum() / mask.sum()
            life_l = (life_l_per * mask).sum() / mask.sum()
        else:
            mdn_l = mdn_l_per.mean()
            life_l = life_l_per.mean()

        total = mdn_l + self.hps.done_loss_scale * life_l
        return total, mdn_l, life_l

    # ------------------------------------------------------------------
    # Single-step sampling (unchanged, just uses done_logits as life-loss)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_next_z(self, logmix, mean, logstd, temperature=1.0):
        B, D, K = logmix.shape

        # adjust mixture logits by temperature
        logmix = logmix / temperature
        logmix = logmix - torch.logsumexp(logmix, dim=-1, keepdim=True)
        mix_probs = torch.exp(logmix)  # [B,D,K]

        mix_idx = torch.distributions.Categorical(mix_probs).sample()  # [B,D]

        mean_chosen = torch.gather(mean, 2, mix_idx.unsqueeze(-1)).squeeze(-1)
        logstd_chosen = torch.gather(logstd, 2, mix_idx.unsqueeze(-1)).squeeze(-1)

        std_chosen = torch.exp(logstd_chosen)
        eps = torch.randn_like(std_chosen) * math.sqrt(temperature)
        z_sample = mean_chosen + std_chosen * eps  # [B,D]
        return z_sample

    @torch.no_grad()
    def step(self, z_t, a_t, h_c_prev=None, temperature=1.0):
        """
        Single-step prediction for dreaming:

        z_t: [B, z_size]
        a_t: [B, action_size]
        h_c_prev: (h_prev, c_prev) or None

        Returns:
          z_{t+1} sample [B,z_size],
          life_loss_prob [B],
          (h_next, c_next)
        """
        B = z_t.shape[0]
        z_t = z_t.to(self.device)
        a_t = a_t.to(self.device)

        if h_c_prev is None:
            h_prev = torch.zeros(1, B, self.rnn_size, device=self.device)
            c_prev = torch.zeros(1, B, self.rnn_size, device=self.device)
        else:
            h_prev, c_prev = h_c_prev

        x = torch.cat([z_t, a_t], dim=-1).unsqueeze(1)  # [B,1,z+A]
        lstm_out, (h_next, c_next) = self.lstm(x, (h_prev, c_prev))  # [B,1,H]

        mdn_out = self.mdn_linear(lstm_out)  # [B,1,z*K*3]
        mdn_out = mdn_out.view(B, 1, self.z_size, self.num_mixture * 3)
        logmix, mean, logstd = torch.split(mdn_out, self.num_mixture, dim=-1)
        logmix = F.log_softmax(logmix, dim=-1)

        # drop time dim
        logmix = logmix[:, 0]   # [B,z,K]
        mean = mean[:, 0]
        logstd = logstd[:, 0]

        z_next = self.sample_next_z(logmix, mean, logstd, temperature=temperature)

        done_logits = self.done_linear(lstm_out).squeeze(-1)  # [B,1] → [B]
        life_loss_prob = torch.sigmoid(done_logits[:, 0])

        return z_next, life_loss_prob, (h_next, c_next)


# ---------------------------------------------------------------------
# Helper functions similar to TF utils (rnn_output etc.)
# ---------------------------------------------------------------------
def rnn_output_size(z_size: int, rnn_size: int, mode: int) -> int:
    """
    Adapted from TF code, but now parameterized by z_size and rnn_size.

    Differences:
      - CHANGED: uses z_size=128, rnn_size=512 for Atari instead of hard-coded 32/256.
    """
    if mode == MODE_ZCH:
        return z_size + 2 * rnn_size
    if mode in (MODE_ZC, MODE_ZH):
        return z_size + rnn_size
    return z_size  # MODE_Z or MODE_Z_HIDDEN


def rnn_output(z, h, c, mode: int):
    """
    Build controller input from z and LSTM state (h, c), like in original code.

    z: [B, z_size]
    h: [1, B, rnn_size]
    c: [1, B, rnn_size]
    """
    if mode == MODE_ZCH:
        return torch.cat([z, c[0], h[0]], dim=-1)
    if mode == MODE_ZC:
        return torch.cat([z, c[0]], dim=-1)
    if mode == MODE_ZH:
        return torch.cat([z, h[0]], dim=-1)
    return z