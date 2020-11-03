from typing import Dict, DefaultDict, Union, Any

import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import torch
from torch import Tensor, optim, nn
from torch.optim import optimizer
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset

import contrastlib

try:
    import tensorboardX as tb
    import tqdm

    IS_SUCCESSFUL = True
except ImportError:
    IS_SUCCESSFUL = False


@dataclasses.dataclass
class Config:
    batch_size: int = 64
    max_steps: int = 2
    test_interval: int = 2
    save_interval: int = 2
    max_grad_value: float = 5.0
    max_grad_norm: float = 100.0
    logdir: Union[str, os.PathLike] = "./logs/"
    gpus: str = ""


class BaseTrainer:
    """Trainer class for ML models.

    Args:
        batch_size: Batch size for training and testing.
        max_steps: Max number of training steps.
        test_interval: Interval steps for testing.
        save_interval: Interval steps for saving checkpoints.
        max_grad_value: Max gradient value.
        max_grad_norm: Max gradient norm.
        logdir: Path to log directory.
        gpus: GPU options.
    """

    def __init__(self, **kwargs: Any) -> None:

        self._config = Config(**kwargs)
        self._global_steps = 0
        self._postfix: Dict[str, float] = {}
        self._loss_key = "loss"

        self._model: nn.Module
        self._logdir: pathlib.Path
        self._logger: logging.Logger
        self._writer: tb.SummaryWriter
        self._train_loader_pos: dataloader.DataLoader
        self._train_loader_neg: dataloader.DataLoader
        self._test_loader_pos: dataloader.DataLoader
        self._test_loader_neg: dataloader.DataLoader
        self._optimizer: optimizer.Optimizer
        self._device: torch.device
        self._pbar: tqdm.tqdm

        if not IS_SUCCESSFUL:
            raise ImportError("Extra requires are not installed.")

    def run(
        self,
        model: nn.Module,
        train_data_pos: Dataset,
        train_data_neg: Dataset,
        test_data_pos: Dataset,
        test_data_neg: Dataset,
    ) -> None:
        """Main run method.

        Args:
            model: ML model.
            train_data_pos: Dataset of positive data for training,
            train_data_neg: Dataset of negative data for training.
            test_data_pos: Dataset of positive data for test.
            test_data_neg: Dataset of negative data for test.
        """

        self._make_logdir()
        self._init_logger()
        self._init_writer()

        try:
            self._logger.info("Start experiment")
            self._logger.info(f"Logdir: {self._logdir}")
            self._logger.info(f"Params: {self._config}")

            self._set_model(model)
            self._set_data(train_data_pos, train_data_neg, test_data_pos, test_data_neg)
            self._start_run()

            self._logger.info("Finish experiment")
        except Exception as e:
            self._logger.exception(f"Run function error: {e}")
            raise RuntimeError(f"Run function error: {e}") from e
        finally:
            self._quit()

    def _make_logdir(self) -> None:

        self._logdir = pathlib.Path(self._config.logdir, time.strftime("%Y%m%d%H%M"))
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _init_logger(self) -> None:

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s"
        )
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(filename=self._logdir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s"
        )
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

        self._logger = logger

    def _init_writer(self) -> None:

        self._writer = tb.SummaryWriter(str(self._logdir))

    def _set_data(
        self,
        train_data_pos: Dataset,
        train_data_neg: Dataset,
        test_data_pos: Dataset,
        test_data_neg: Dataset,
    ) -> None:

        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        # Params for GPU
        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self._train_loader_pos = torch.utils.data.DataLoader(  # type: ignore
            train_data_pos, shuffle=True, batch_size=self._config.batch_size, **kwargs
        )
        self._train_loader_neg = torch.utils.data.DataLoader(  # type: ignore
            train_data_neg, shuffle=True, batch_size=self._config.batch_size, **kwargs
        )

        self._test_loader_pos = torch.utils.data.DataLoader(  # type: ignore
            test_data_pos, shuffle=False, batch_size=self._config.batch_size, **kwargs
        )
        self._test_loader_neg = torch.utils.data.DataLoader(  # type: ignore
            test_data_neg, shuffle=True, batch_size=self._config.batch_size, **kwargs
        )

        assert len(self._train_loader_pos) == len(self._train_loader_neg)
        assert len(self._test_loader_pos) == len(self._test_loader_neg)

        self._logger.info(f"Train dataset size: {len(self._train_loader_pos)}")
        self._logger.info(f"Test dataset size: {len(self._test_loader_pos)}")

    def _start_run(self) -> None:

        if self._config.gpus:
            self._device = torch.device(f"cuda:{self._config.gpus}")
        else:
            self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        self._pbar = tqdm.tqdm(total=self._config.max_steps)
        self._global_steps = 0
        self._postfix = {"train/loss": 0.0, "test/loss": 0.0}

        while self._global_steps < self._config.max_steps:
            self._train()

        self._pbar.close()

    def _train(self) -> None:

        for (data_pos, _), (data_neg, _) in zip(self._train_loader_pos, self._train_loader_neg):
            self._model.train()
            self._optimizer.zero_grad()

            data_pos = data_pos.to(self._device)
            data_neg = data_neg.to(self._device)
            loss_dict = self._train_step(data_pos, data_neg)
            loss = loss_dict[self._loss_key].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self._model.parameters(), self._config.max_grad_value)
            self._optimizer.step()

            loss_dict_second = self._train_second_step(data_pos, data_neg)
            loss_dict.update(loss_dict_second)

            self._train_step_end()

            self._global_steps += 1
            self._pbar.update(1)

            self._postfix["train/loss"] = loss_dict[self._loss_key].mean().item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                self._writer.add_scalar(f"train/{key}", value.mean(), self._global_steps)

            if self._global_steps % self._config.test_interval == 0:
                self._test()

            if self._global_steps % self._config.save_interval == 0:
                self._save_checkpoint()
                self._save_plots()

                loss_logger = {k: v.mean() for k, v in loss_dict.items()}
                self._logger.debug(f"Train loss (steps={self._global_steps}): " f"{loss_logger}")

            if self._global_steps >= self._config.max_steps:
                break

    def _test(self) -> None:

        loss_logger: DefaultDict[str, float] = collections.defaultdict(float)
        self._model.eval()
        for (data_pos, _), (data_neg, _) in zip(self._test_loader_pos, self._test_loader_neg):
            with torch.no_grad():
                data_pos = data_pos.to(self._device)
                data_neg = data_neg.to(self._device)
                loss_dict = self._test_step(data_pos, data_neg)

            self._postfix["test/loss"] = loss_dict[self._loss_key].mean().item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                loss_logger[key] += value.sum().item()

        for key, value in loss_logger.items():  # type: ignore
            self._writer.add_scalar(
                f"test/{key}",
                value / (len(self._test_loader_pos)),
                self._global_steps,
            )

        self._logger.debug(f"Test loss (steps={self._global_steps}): {loss_logger}")

    def _save_checkpoint(self) -> None:

        # Remove unused prefix
        model_state_dict = {}
        for k, v in self._model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self._optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        state_dict = {
            "steps": self._global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        path = self._logdir / f"checkpoint_{self._global_steps}.pt"
        torch.save(state_dict, path)
        self._logger.debug(f"Saved checkpoint {str(path)}.")

    def _quit(self) -> None:

        self._save_configs()
        self._writer.close()

    def _save_configs(self) -> None:

        config = dataclasses.asdict(self._config)
        config["logdir"] = str(self._logdir)
        path = self._logdir / "config.json"
        with path.open("w") as f:
            json.dump(config, f)
        self._logger.debug(f"Saved config file {str(path)}.")

    def _set_model(self, model: nn.Module) -> None:

        raise NotImplementedError

    def _train_step(self, data: Tensor, label: Tensor) -> Dict[str, Tensor]:

        raise NotImplementedError

    def _test_step(self, data: Tensor, label: Tensor) -> Dict[str, Tensor]:

        raise NotImplementedError

    def _train_second_step(self, data: Tensor, label: Tensor) -> Dict[str, Tensor]:

        return {}

    def _train_step_end(self) -> None:

        return

    def _save_plots(self) -> None:

        return


class Trainer(BaseTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._model: contrastlib.BaseModel

    def _set_model(self, model: nn.Module) -> None:

        assert isinstance(model, contrastlib.BaseModel)
        self._model = model
        self._optimizer = optim.Adam(self._model.parameters())

    def _train_step(self, data: Tensor, label: Tensor) -> Dict[str, Tensor]:

        return self._model.loss_func(x_p=data, x_n=label)

    def _test_step(self, data: Tensor, label: Tensor) -> Dict[str, Tensor]:

        return self._model.loss_func(x_p=data, x_n=label)
