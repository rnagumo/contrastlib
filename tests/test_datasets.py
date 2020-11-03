import contrastlib


def test_dataset() -> None:
    data_num = 4
    seq_len = 10
    dataset = contrastlib.SequentialMNIST(
        data_num=data_num, seq_len=seq_len, root="../data/mnist", train=True, download=False
    )

    data, target = dataset[[0, 1]]  # type: ignore
    assert data.size() == (2, 10, 3, 64, 64)
    assert (0 <= data).all() and (data <= 1).all()
    assert target.size() == (2, 10)
    assert len(dataset) == 4


def test_dataset_colored() -> None:
    data_num = 4
    seq_len = 10
    dataset = contrastlib.SequentialMNIST(
        data_num=data_num,
        seq_len=seq_len,
        color=True,
        root="../data/mnist",
        train=True,
        download=False,
    )

    data, target = dataset[[0, 1]]  # type: ignore
    assert data.size() == (2, 10, 3, 64, 64)
    assert (0 <= data).all() and (data <= 1).all()
    assert target.size() == (2, 10)
    assert len(dataset) == 4
