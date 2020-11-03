from typing import Dict

from torch import Tensor, nn


class BaseModel(nn.Module):
    """Base class for contrastive learning."""

    def forward(self, x: Tensor) -> Tensor:
        """Encodes observations to context `c`.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            c (torch.Tensor): Encoded context, size `(b, d)`.
        """

        raise NotImplementedError

    def loss_func(self, x_p: Tensor, x_n: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x_p (torch.Tensor): Positive observations, size `(b, c, h, w)`.
            x_n (torch.Tensor): Negative observations, size `(b, c, h, w)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError
