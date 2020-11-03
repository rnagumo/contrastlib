import torch
import contrastlib


def test_cpc_forward() -> None:
    model = contrastlib.ContrastivePredictiveModel()
    x = torch.rand(8, 6, 3, 64, 64)
    c = model(x)

    assert c.size() == (8, 6, 10)


def test_cpc_loss_func() -> None:
    model = contrastlib.ContrastivePredictiveModel()
    x_p = torch.rand(8, 6, 3, 64, 64)
    x_n = torch.rand(8, 6, 3, 64, 64)
    loss_dict = model.loss_func(x_p, x_n)

    assert isinstance(loss_dict, dict)
    assert not torch.isnan(loss_dict["loss"]).any()
