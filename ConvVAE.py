import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE84x84(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), # 84 -> 42
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 42 -> 21
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),# 21 -> 10
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),# 10 -> 5
            nn.ReLU(inplace=True),
        )

        self.enc_fc = nn.Linear(5 * 5 * 256, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # ----- Decoder -----
        self.dec_fc = nn.Linear(latent_dim, 1024)
        self.dec_fc2 = nn.Linear(1024, 5 * 5 * 256)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),             # 5 -> 10
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1),   # 10 -> 21
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                  # 21 -> 42
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),                       # 42 -> 84
            nn.Sigmoid(),   # if inputs are scaled to [0,1]
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 256, 5, 5)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    
class ConvVAE(object):
    
    def __init__(
        self,
        z_size=32,
        batch_size=1,
        learning_rate=0.0001,
        kl_tolerance=0.5,
        is_training=False,
        reuse=False,   # kept for API compatibility, no-op
        gpu_mode=False
    ):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse  # unused in PyTorch, but kept for interface

        # ----- Device selection -----
        if gpu_mode and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ----- Model -----
        self.model = ConvVAE84x84(latent_dim=z_size).to(self.device)
        
        # ----- Optimizer  -----
        if self.is_training:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        else:
            self.optimizer = None
            
            
    
    def _np_to_torch_x(self, x_np):
        """
        Convert NHWC numpy [N, 84, 84, 1] to NCHW torch [N, 1, 84, 84]
        and move to device.
        """
        x = torch.from_numpy(x_np).float()        # (N,84,84,1)
        x = x.permute(0, 3, 1, 2)                # -> (N,1,84,84)
        return x.to(self.device)
    
    def _torch_to_np_x(self, x_torch):
        """
        Convert NCHW torch [N,1,84,84] to NHWC numpy [N,84,84,1].
        """
        x = x_torch.detach().cpu()
        x = x.permute(0, 2, 3, 1)  # (N,84,84,1)
        return x.numpy()
    
    def encode(self, x):
        """
        Like TF: returns z (sampled) for input x.
        x: numpy, shape [N,84,84,1], float32 in [0,1]
        returns z: numpy, shape [N,z_size]
        """
        self.model.eval()
        with torch.no_grad():
            x_t = self._np_to_torch_x(x)
            mu, logvar = self.model.encode(x_t)
            z = self.model.reparameterize(mu, logvar)
        return z.cpu().numpy()
    
    def encode_mu_logvar(self, x):
        """
        Return (mu, logvar) without sampling.
        x: numpy [N,84,84,1]
        returns: (mu_np, logvar_np), both [N,z_size]
        """
        self.model.eval()
        with torch.no_grad():
            x_t = self._np_to_torch_x(x)
            mu, logvar = self.model.encode(x_t)
        return mu.cpu().numpy(), logvar.cpu().numpy()
    
    def decode(self, z):
        """
        Decode latent z to images.
        z: numpy [N,z_size]
        returns: y_np [N,84,84,1] in [0,1]
        """
        self.model.eval()
        z_t = torch.from_numpy(z).float().to(self.device)
        with torch.no_grad():
            y_t = self.model.decode(z_t)    # (N,1,84,84)
        return self._torch_to_np_x(y_t)
    
    
    def _compute_loss(self, x_t):
        """
        x_t: torch tensor [N,1,84,84] in [0,1]
        Returns (loss, r_loss, kl_loss)
        """
        x_recon, mu, logvar = self.model(x_t)

        # reconstruction loss (MSE, like tf.reduce_sum square then mean)
        # Per-sample sum over pixels
        r_loss = torch.sum((x_t - x_recon) ** 2, dim=(1, 2, 3))
        r_loss = torch.mean(r_loss)

        # KL loss per dim, same formula:
        # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        # Apply KL tolerance per sample
        kl_min = self.kl_tolerance * self.z_size
        kl_loss = torch.clamp(kl_loss, min=kl_min)
        kl_loss = torch.mean(kl_loss)

        loss = r_loss + kl_loss
        return loss, r_loss, kl_loss
    
    
    def train_on_batch(self, x):
        """
        Single optimization step on one batch.
        x: numpy [N,84,84,1] in [0,1]
        returns: (loss, r_loss, kl_loss) as Python floats
        """
        if not self.is_training:
            raise RuntimeError("ConvVAE was initialized with is_training=False")

        self.model.train()
        x_t = self._np_to_torch_x(x)

        self.optimizer.zero_grad()
        loss, r_loss, kl_loss = self._compute_loss(x_t)
        loss.backward()
        self.optimizer.step()

        return (
            float(loss.detach().cpu().item()),
            float(r_loss.detach().cpu().item()),
            float(kl_loss.detach().cpu().item()),
        )
        
    def get_model_params(self):
        """
        Similar to TF version:
          returns (model_params, model_shapes, model_names)
        where model_params is a list of int lists (quantized *10000).
        """
        model_params = []
        model_shapes = []
        model_names = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                p = param.detach().cpu().numpy()
                model_names.append(name)
                model_shapes.append(p.shape)
                q = np.round(p * 10000).astype(np.int32).tolist()
                model_params.append(q)

        return model_params, model_shapes, model_names
    
    def get_random_model_params(self, stdev=0.5):
        """
        Same spirit as TF: draw random params from standard_cauchy * stdev.
        Returns a list of numpy arrays matching parameter shapes.
        """
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            rparam.append(np.random.standard_cauchy(s) * stdev)
        return rparam
    
    def set_model_params(self, params):
        """
        Set model parameters from quantized format (list of lists),
        like TF's set_model_params.
        If you pass the output of get_random_model_params, you should skip
        quantization and just assign directly (see set_random_params).
        """
        # Convert flat params (lists) back to np arrays and divide by 10000
        with torch.no_grad():
            idx = 0
            for _, param in self.model.named_parameters():
                pshape = param.shape
                p_list = params[idx]
                p_np = np.array(p_list, dtype=np.float32) / 10000.0
                p_np = p_np.reshape(pshape)
                param.copy_(torch.from_numpy(p_np))
                idx += 1
                
                
    def set_random_params(self, stdev=0.5):
        """
        Equivalent of TF set_random_params: sample random params with
        Cauchy distribution * stdev and assign them directly.
        """
        rand_params = self.get_random_model_params(stdev)
        with torch.no_grad():
            idx = 0
            for _, param in self.model.named_parameters():
                p_np = rand_params[idx].astype(np.float32)
                assert p_np.shape == tuple(param.shape)
                param.copy_(torch.from_numpy(p_np))
                idx += 1
                
    def save_json(self, jsonfile='vae.json'):
        """
        Save quantized parameters (like TF save_json) into a JSON file.
        """
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'w') as f:
            json.dump(qparams, f, sort_keys=True, indent=0, separators=(',', ': '))
            
            
    def load_json(self, jsonfile='vae.json'):
        """
        Load quantized parameters from JSON and assign to model.
        """
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
        
        
    def save_model(self, model_save_path):
        """
        Save a binary PyTorch checkpoint (instead of TF .ckpt).
        """
        os.makedirs(model_save_path, exist_ok=True)
        checkpoint_path = os.path.join(model_save_path, 'vae.pt')
        print(f"Saving model to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)
        
        
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load weights from a PyTorch checkpoint path, e.g. 'models/vae.pt'.
        """
        print(f"Loading model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
