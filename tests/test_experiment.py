from typing import Tuple, Dict

from copy import deepcopy
import tempfile

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import contrastlib
from contrastlib.base import BaseModel


def test_trainer_run() -> None:

    model = TempModel()
    train_data_pos = TempDataset()
    train_data_neg = TempDataset()
    test_data_pos = TempDataset()
    test_data_neg = TempDataset()

    org_params = deepcopy(model.state_dict())

    with tempfile.TemporaryDirectory() as logdir:
        trainer = contrastlib.Trainer(logdir=logdir)
        trainer.run(model, train_data_pos, train_data_neg, test_data_pos, test_data_neg)

        root = trainer._logdir
        assert (root / "training.log").exists()
        assert (root / "config.json").exists()

    updated_params = model.state_dict()
    for key in updated_params:
        assert not (updated_params[key] == org_params[key]).all()


class TempModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Linear(64 * 64 * 3, 10, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        return self.encoder(x.view(-1, 64 * 64 * 3))

    def loss_func(self, x_p: Tensor, x_n: Tensor) -> Dict[str, Tensor]:

        z_p = self.encoder(x_p.view(-1, 64 * 64 * 3))
        z_n = self.encoder(x_n.view(-1, 64 * 64 * 3))
        loss_dict = {"loss": F.mse_loss(z_p, z_n)}

        return loss_dict


class TempDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data = torch.rand(10, 3, 64, 64)
        self._label = torch.randint(0, 100, (10,))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._data[index], self._label[index]

    def __len__(self) -> int:
        return self._data.size(0)
