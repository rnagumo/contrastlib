"""Dataset class for sequential MNIST.

Ref)
https://github.com/davidtellez/contrastive-predictive-coding/blob/master/data_utils.py
"""

from typing import List, Tuple, Any

import random

from PIL import Image

from sklearn.datasets import load_sample_image

import torch
from torch import Tensor
from torchvision import datasets, transforms


class SequentialMNIST(datasets.MNIST):
    """Sequential MNIST dataset.

    This dataset contains 3-channels MNIST images.

    Args:
        data_num (int): Number of sequences.
        seq_len (int): Length of each sequence.
        ordered (bool, optional): If `True`, sequence is ordered numbers.
        color (bool, optional): If `True`, coloring pixels.
        image_name (str, optional): Background image name for coloring.

    Attributes:
        indices (torch.Tensor): Indices for sequences.
    """

    def __init__(
        self,
        data_num: int,
        seq_len: int,
        ordered: bool = False,
        color: bool = False,
        image_name: str = "china.jpg",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.data_num = data_num
        self.seq_len = seq_len
        self.ordered = ordered
        self.color = color

        self._preprocess_data(image_name)
        self.indices = torch.tensor([self._generate_indices() for _ in range(data_num)])

    def _preprocess_data(self, image_name: str) -> None:

        _transform_mnist = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
        _transform_background = transforms.Compose(
            [transforms.RandomCrop(64), transforms.ToTensor()]
        )

        if self.color:
            background_image = Image.fromarray(load_sample_image(image_name))

        data_list = []
        for img in self.data:  # type: ignore
            img = Image.fromarray(img.numpy(), mode="L")
            img = _transform_mnist(img)
            img = img.repeat(3, 1, 1)

            if self.color:
                img[img >= 0.5] = 1.0
                img[img < 0.5] = 0.0

                color_img = _transform_background(background_image)
                color_img = (color_img + torch.rand(3, 1, 1)) / 2
                color_img[img == 1] = 1 - color_img[img == 1]
                img = color_img

            data_list.append(img)

        self.data = torch.stack(data_list)

    def _generate_indices(self) -> List[int]:

        n = random.randint(0, 9)
        indices: List[int] = []
        for _ in range(self.seq_len):
            t, *_ = torch.where(self.targets == n)
            idx = t[torch.multinomial(t.float(), 1)].item()
            assert isinstance(idx, int)
            indices.append(idx)

            if self.ordered:
                n = (n + 1) % 10
            else:
                n = random.randint(0, 9)

        return indices

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get sequence.

        Args:
            index (int): Index of data from batch.

        Returns:
            img (torch.Tensor): Sequence of images, size `(l, c, h, w)`.
            target (torch.Tensor): Sequence of targets, size `(l,)`.
        """

        img = self.data[self.indices[index]]
        target = self.targets[self.indices[index]]

        return img, target

    def __len__(self) -> int:
        """Number of batch.

        Returns:
            data_num (int): Number of sequences.
        """

        return self.data_num
