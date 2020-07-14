
"""Contrastive Predictive Coding."""

from typing import Dict

import torch
from torch import Tensor, nn

from .base import BaseModel


class Encoder(nn.Module):
    """Encoder: z = f(x).

    Args:
        in_channels (int): Channel size of inputs.
        z_dim (int): Dimension size of latents.
    """

    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes observations.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, z)`.
        """

        h = self.conv(x)
        h = h.view(-1, 1024)
        z = self.fc(h)

        return z


class DenstiyRatioEstimator(nn.Module):
    """Density ratio estimation function f(x, c).

    Args:
        z_dim (int): Dimension size of latents.
        c_dim (int): Dimension size of contexts.
        predictive_steps (int): Number of time steps for training.
    """

    def __init__(self, z_dim: int, c_dim: int, predictive_steps: int):
        super().__init__()

        self.predictive_steps = predictive_steps
        self.trans = nn.Parameter(torch.zeros(predictive_steps, z_dim, c_dim))

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        """Computes density ratio.

        Args:
            z (torch.Tensor): Latent representations, size `(b, l, z)`.
            c (torch.Tensor): Context representations, size `(b, l, c)`.

        Returns:
            r (torch.Tensor): Density ratio, size `(b, l)`.
        """

        # Dataset size
        batch, seq_len, *_ = z.size()

        if seq_len > self.predictive_steps:
            raise ValueError(
                f"Given sequence length ({seq_len}) must be equal or smaller "
                f"than predictive steps ({self.predictive_steps}).")

        # Calculate
        r = z.new_zeros((batch, seq_len))
        for t in range(seq_len):
            r[:, t] = (
                z[:, t].matmul(self.trans[t].matmul(c[:, t].t()))).sum(-1)

        return r.exp()


class ContrastivePredictiveModel(BaseModel):
    """Contrastive Predictive Coding.

    Args:
        in_channels (int, optional): Channel size of inputs.
        z_dim (int, optional): Dimension size of latents.
        c_dim (int, optional): Dimension size of contexts.
        predictive_steps (int, optional): Number of time steps for training.
    """

    def __init__(self, in_channels: int = 3, z_dim: int = 10, c_dim: int = 10,
                 predictive_steps: int = 2):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictive_steps = predictive_steps

        self.encoder = Encoder(in_channels, z_dim)
        self.rnn_cell = nn.GRUCell(z_dim, c_dim)
        self.estimator = DenstiyRatioEstimator(z_dim, c_dim, predictive_steps)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes observations to context `c`.

        Args:
            x (torch.Tensor): Observations, size `(b, l, c, h, w)`.

        Returns:
            c (torch.Tensor): Encoded context, size `(b, l, d)`.
        """

        # Data size
        batch, seq_len, *_ = x.size()

        # Encode observations
        z_p = x.new_zeros((batch, seq_len, self.z_dim))
        for t in range(seq_len):
            z_p[:, t] = self.encoder(x[:, t])

        # Produce context latent representations
        c_t = x.new_zeros((batch, self.c_dim))
        c = x.new_zeros((batch, seq_len, self.c_dim))
        for t in range(seq_len):
            c_t = self.rnn_cell(z_p[:, t], c_t)
            c[:, t] = c_t

        return c

    def loss_func(self, x_p: Tensor, x_n: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x_p (torch.Tensor): Positive observations, size `(b, l, c, h, w)`.
            x_n (torch.Tensor): Negative observations, size `(b, l, c, h, w)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        # Data size
        batch, seq_len, *_ = x_p.size()

        # Encode observations
        z_p = x_p.new_zeros((batch, seq_len, self.z_dim))
        z_n = x_n.new_zeros((batch, seq_len, self.z_dim))
        for t in range(seq_len):
            z_p[:, t] = self.encoder(x_p[:, t])
            z_n[:, t] = self.encoder(x_n[:, t])

        # Produce context latent representations
        c_t = x_p.new_zeros((batch, self.c_dim))
        c = x_p.new_zeros((batch, seq_len, self.c_dim))
        for t in range(seq_len):
            c_t = self.rnn_cell(z_p[:, t], c_t)
            c[:, t] = c_t

        # Calculate log density ratio for positive and negative samples
        l_n = x_p.new_zeros((batch,))
        for t in range(seq_len - 1):
            r_p = self.estimator(z_p[:, t:t + self.predictive_steps],
                                 c[:, t:t + self.predictive_steps])
            r_n = self.estimator(z_n[:, t:t + self.predictive_steps],
                                 c[:, t:t + self.predictive_steps])
            l_n += (r_p.log() - r_n.log()).sum(1)

        loss = -l_n.mean()

        return {"loss": loss}
