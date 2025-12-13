import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Modes (for controller input, like original code)
# ---------------------------------------------------------------------
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4

# ---------------------------------------------------------------------
# Hyperparameters (PyTorch version of HyperParams)
# ---------------------------------------------------------------------
@dataclass
class HyperParams:
    num_steps: int = 3000
    max_seq_len: int = 200          # full sequence length (we predict up to T-1 -> t+1)
    input_seq_width: int = 128 + 4   # z_size + action_size
    output_seq_width: int = 128      # z_size
    rnn_size: int = 512
    batch_size: int = 64
    grad_clip: float = 1.0
    num_mixture: int = 5
    learning_rate: float = 1e-3
    decay_rate: float = 1.0          # if you want, can do manual decay
    min_learning_rate: float = 1e-5
    use_layer_norm: int = 0
    use_recurrent_dropout: int = 0
    recurrent_dropout_prob: float = 0.9
    use_input_dropout: int = 0
    input_dropout_prob: float = 0.9
    use_output_dropout: int = 0
    output_dropout_prob: float = 0.9
    is_training: int = 1

    # Atari-specific extras (same info as TF version but explicit)
    z_size: int = 128
    action_size: int = 4


def default_hps() -> HyperParams:
    return HyperParams()


# ---------------------------------------------------------------------
# MDN-RNN in PyTorch (TF logic, but no life-loss / done head)
# ---------------------------------------------------------------------
class MDNRNN(nn.Module):
    """
    MDN-RNN for Atari Breakout (PyTorch version of your TF MDNRNN),
    with ONLY:
      - MDN prediction for z_{t+1} given (z_t, a_t)
    No life-loss / done head.
    """

    def __init__(self, hps: HyperParams, device: torch.device | None = None):
        super().__init__()
        self.hps = hps
        self.z_size = hps.z_size
        self.action_size = hps.action_size
        self.num_mixture = hps.num_mixture
        self.rnn_size = hps.rnn_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Input is concat([z_t, a_t]) like in the TF version
        input_size = self.z_size + self.action_size
        output_z = self.z_size  # predict z_{t+1}

        # LSTM backbone (like tf.keras.layers.LSTM with return_sequences=True)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.rnn_size,
            num_layers=1,
            batch_first=True,
        )

        # MDN head: for each z-dim, we output K mixtures: (logmix, mean, logstd)
        self.mdn_linear = nn.Linear(
            self.rnn_size, output_z * self.num_mixture * 3
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward pass for an entire sequence
    # ------------------------------------------------------------------
    def forward(self, z_seq, a_seq, h0=None, c0=None):
        """
        z_seq: [B, T, z_size]       (z_t, full sequence)
        a_seq: [B, T, action_size]  (a_t, full sequence)

        We will typically use:
          - inputs  = z[:, :T-1], a[:, :T-1]
          - targets = z[:, 1:  ]
        outside of this function.

        Returns:
          logmix: [B, T, z_size, K]
          mean:   [B, T, z_size, K]
          logstd: [B, T, z_size, K]
          (h_n, c_n): final LSTM state
        """
        B, T, _ = z_seq.shape

        x = torch.cat([z_seq, a_seq], dim=-1)  # [B,T,z+A]

        if h0 is None or c0 is None:
            h0 = torch.zeros(1, B, self.rnn_size, device=self.device)
            c0 = torch.zeros(1, B, self.rnn_size, device=self.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))  # lstm_out: [B,T,H]

        # MDN head
        mdn_out = self.mdn_linear(lstm_out)            # [B,T,z*K*3]
        mdn_out = mdn_out.view(B, T, self.z_size, self.num_mixture * 3)

        logmix, mean, logstd = torch.split(mdn_out, self.num_mixture, dim=-1)
        # mixture weights normalized over mixture dim
        logmix = F.log_softmax(logmix, dim=-1)

        return logmix, mean, logstd, (h_n, c_n)

    # ------------------------------------------------------------------
    # MDN loss per timestep (like TF's tf_lognormal + mixture sum)
    # ------------------------------------------------------------------
    def mdn_loss_per_timestep(self, z_target, logmix, mean, logstd):
        """
        z_target: [B, T, z]
        logmix, mean, logstd: [B, T, z, K]

        Returns:
          nll_per_timestep: [B, T]
        """
        # y: [B,T,z,1]
        y = z_target.unsqueeze(-1)

        log_two_pi = math.log(2.0 * math.pi)

        # log N(y | mean, std)
        inv_std = torch.exp(-logstd)
        normed = (y - mean) * inv_std
        log_prob = -0.5 * (normed ** 2 + 2.0 * logstd + log_two_pi)  # [B,T,z,K]

        # add mixture weights (logmix) and log-sum-exp over K
        log_prob = logmix + log_prob  # [B,T,z,K]
        log_prob = torch.logsumexp(log_prob, dim=-1)  # [B,T,z]

        # negative log-likelihood averaged over z dimension
        nll = -log_prob.mean(dim=-1)  # [B,T]
        return nll

    # ------------------------------------------------------------------
    # Sequence loss (z-only, no done)
    # ------------------------------------------------------------------
    def sequence_loss(self, z_full, a_full, mask=None):
        """
        Compute sequence MDN loss for z_{t+1}.

        z_full: [B, T, z_size]       (entire sequence z_0..z_{T-1})
        a_full: [B, T, action_size]  (entire sequence a_0..a_{T-1})
        mask:   [B, T-1] optional, 1=valid, 0=padding

        Uses:
          inputs  = (z_t, a_t)   for t = 0..T-2
          targets = z_{t+1}      for t = 0..T-2
        """
        z_full = z_full.to(self.device)
        a_full = a_full.to(self.device)

        # T-1 inputs -> predict z_{t+1}
        z_t   = z_full[:, :-1, :]  # [B, T-1, z]
        a_t   = a_full[:, :-1, :]  # [B, T-1, A]
        z_tp1 = z_full[:, 1:, :]   # [B, T-1, z]

        logmix, mean, logstd, _ = self.forward(z_t, a_t)
        # shapes: [B, T-1, z, K]

        mdn_l_per = self.mdn_loss_per_timestep(z_tp1, logmix, mean, logstd)  # [B, T-1]

        if mask is not None:
            mask = mask.to(self.device)
            loss = (mdn_l_per * mask).sum() / mask.sum()
        else:
            loss = mdn_l_per.mean()

        return loss

    # ------------------------------------------------------------------
    # Single-step sampling (for dreaming; z-only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_next_z(self, logmix, mean, logstd, temperature=1.0):
        """
        logmix, mean, logstd: [B, z, K]  (no time dimension)
        Returns:
          z_sample: [B, z]
        """
        B, D, K = logmix.shape

        # temperature scaling on mixture logits
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
        Single-step rollout:
          z_t: [B, z_size]
          a_t: [B, action_size]
        Returns:
          z_{t+1} sample [B, z_size],
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

        # drop time dimension
        logmix = logmix[:, 0]  # [B,z,K]
        mean   = mean[:, 0]
        logstd = logstd[:, 0]

        z_next = self.sample_next_z(logmix, mean, logstd, temperature=temperature)
        return z_next, (h_next, c_next)


# ---------------------------------------------------------------------
# Helper functions like original rnn_output / rnn_output_size
# ---------------------------------------------------------------------
def rnn_output_size(z_size: int, rnn_size: int, mode: int) -> int:
    """
    Same semantics as in TF/ES code, but parameterized by z_size/rnn_size.
    """
    if mode == MODE_ZCH:
        return z_size + 2 * rnn_size
    if mode in (MODE_ZC, MODE_ZH):
        return z_size + rnn_size
    return z_size  # MODE_Z or MODE_Z_HIDDEN


def rnn_output(z, h, c, mode: int):
    """
    Build controller input from z and LSTM state (h, c).
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
