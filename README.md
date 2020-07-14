
# contrastlib

Contrastive learning in PyTorch.

# Requirements

* Python == 3.7
* PyTorch == 1.5.1
* torchvision == 0.6.1
* scikit-learn == 0.23.1

Additional requirements for example codes.

* matplotlib == 3.2.2
* tqdm == 4.46.1
* tensorboardX == 2.0

# Setup

Clone repository.

```bash
git clone https://github.com/rnagumo/contrastlib.git
cd contrastlib
```

Install the package in virtual env.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

Or use [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker). You can run container with GPUs by Docker 19.03+.

```bash
docker build -t contrastlib .
docker run --gpus all -it contrastlib bash
```

Install other requirements for example code.

```bash
pip3 install matplotlib==3.2.2 tqdm==4.46.1 tensorboardX==2.0
```

# Experiments

Run the shell script in `bin` directory. See the script for the experimental detail.

```bash
bash bin/train.sh
```

# References

* A. Oord *et al*., ["Representation Learning with Contrastive Predictive Coding"](http://arxiv.org/abs/1807.03748)
* davidtellez, [contrastive-predictive-coding](https://github.com/davidtellez/contrastive-predictive-coding)
