# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="cce26209"
# # ✔️ Prerequisites

# + [markdown] id="affab565"
# First of all we need to take care of a few **prerequisites**, most notably:
#
# - Install the various pip modules that we will be using.
# - Install some linux specific dependencies of our [content loss](#content-loss).
# - Initialize the Random Number Generator(s), so that our experiments can be replicated.
# - Determine:
#   - The current working directory, as it's going to be used to reference various files such as the dataset, our model checkpoints e.t.c
#   - The available hardware backend. GPU utilization is preferable, as it results in higher complition time.
# - `(Optionally)` Mount Google Drive, where we can load our dataset from.

# + [markdown] id="19e0a6d0"
# ## Determining the Current Working Directory

# + cellView="code" id="945a9ccd"
from pathlib import Path

BASE_DIR = Path.cwd()

# + [markdown] id="94c4d99f"
# ## Mounting Google Drive

# + cellView="code" id="12dcaea0"
GDRIVE_DIR = BASE_DIR / "drive"

# + cellView="code" id="e24f0051"
try:
    from google.colab import drive

    drive.mount(f"{GDRIVE_DIR}")
except ImportError:
    pass

# + [markdown] id="16a902e2"
# ## Installing [graphviz](https://graphviz.org/) & [libgraphviz-dev](https://packages.debian.org/jessie/libgraphviz-dev)

# + [markdown] id="8c0cdd85"
# The aforementioned packages are required by [PyINSECT](https://github.com/billsioros/PyINSECT/tree/implementing-HPGs) and more specifically its graph plotting methods.

# + cellView="code" id="919734ca"
# !sudo apt-get install graphviz libgraphviz-dev

# + [markdown] id="7f5668f4"
# ## Installing the required `pip` modules

# + [markdown] id="aebad6f1"
# - [torch](https://pytorch.org/) is our machine learning framework of choice.
# - [numpy](https://numpy.org/), [sympy](https://www.sympy.org/en/index.html) and [scipy](https://www.scipy.org/) are used to in the context of nanorough surface generation.
# - [plotly](https://plotly.com/) (which requires [pandas](https://pandas.pydata.org/)) as well as [matplotlib](https://matplotlib.org/) are used in order to plot various graphs.

# + cellView="code" id="1057687b"
# !pip install /content/drive/MyDrive/Thesis/roughml-1.0.0-py3-none-any.whl

# + [markdown] id="0192c059"
# ## Initializing (a.k.a `Seeding`) the Random Number Generator(s)

# + [markdown] id="fa44f756"
# We are required to seed various random number generation engines, so that our experiments can be replicated on a later date.

# + cellView="code" id="5daee8da"
SEED = 1234

import os
import random

import numpy as np

# + cellView="code" id="4d6c30c9"
import torch

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)

# + [markdown] id="6fce8a2a"
# ## Determining available backend

# + [markdown] id="5ca328f8"
# By default, we are going to be utilizing the available CPU backend, if no GPU is available.

# + cellView="code" id="ebabd4a5"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

# + cellView="code" id="520ba5c1"
device = torch.device(device)

# + [markdown] id="64c09dfe"
# ## Configuring our Loggers

# + cellView="code" id="5eea1ad8"
import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s", level=logging.CRITICAL
)

# + id="b45114ef"
logger = logging.getLogger()

# + [markdown] id="60d78d1e"
# # 🙃 A naive-approach

# + [markdown] id="ec374650"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="d6594cb1"
from roughml.models import PerceptronGenerator

generator = PerceptronGenerator.from_device(device)

# + cellView="code" id="4bff3a44"
generator

# + cellView="code" id="cac059ee"
from roughml.models import PerceptronDiscriminator

discriminator = PerceptronDiscriminator.from_generator(generator, device=device)

# + cellView="code" id="64022987"
discriminator

# + [markdown] id="eb813599"
# ## Training

# + cellView="code" id="8a131c21"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

# + cellView="code" id="90443487"
from pathlib import Path

CHECKPOINT_DIR = BASE_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# + cellView="code" id="63114157"
from roughml.content.loss import NGramGraphContentLoss
from roughml.data.transforms import Flatten, To
from roughml.training.flow import TrainingFlow
from roughml.training.manager import per_epoch

training_flow = TrainingFlow(
    training_manager={
        "benchmark": True,
        "checkpoint_dir": CHECKPOINT_DIR,
        "checkpoint_multiple": False,
        "train_epoch": per_epoch,
        "log_every_n": 10,
        "criterion": criterion,
        "n_epochs": 10,
        "train_ratio": 0.8,
        "optimizer": {"lr": 0.0005, "weight_decay": 0},
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
        },
    },
    content_loss_type=NGramGraphContentLoss,
    dataset={
        "limit": 10,
        "path": GDRIVE_DIR / "MyDrive" / "Thesis" / "Datasets" / "surfaces.zip",
        "transforms": [Flatten(), To(device)],
    },
)

# + id="836ed418"
training_flow(generator, discriminator)

# + [markdown] id="fe589c1a"
# # 😎 A CNN based approach

# + [markdown] id="b16a4946"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="e301a7a0"
from src.roughml.models import CNNGenerator

generator = CNNGenerator.from_device(device)

# + cellView="code" id="36ffe487"
generator

# + cellView="code" id="5a8a9aad"
from src.roughml.models import CNNDiscriminator

discriminator = CNNDiscriminator.from_device(device)

# + cellView="code" id="1d0a4a40"
discriminator

# + [markdown] id="7bbbcc22"
# ## Training

# + cellView="code" id="45778b4c"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

# + id="QOVvzTEv0o6V"
from pathlib import Path

CHECKPOINT_DIR = BASE_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# + cellView="code" id="82fb12f6"
from roughml.content.loss import ArrayGraph2DContentLoss
from roughml.data.transforms import To, View
from roughml.training.flow import TrainingFlow
from roughml.training.manager import per_epoch

training_flow = TrainingFlow(
    training_manager={
        "benchmark": True,
        "checkpoint_dir": CHECKPOINT_DIR,
        "checkpoint_multiple": False,
        "train_epoch": per_epoch,
        "log_every_n": 10,
        "criterion": criterion,
        "n_epochs": 10,
        "train_ratio": 0.8,
        "optimizer": {"lr": 0.0002, "betas": (0.5, 0.999)},
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
        },
    },
    content_loss_type=ArrayGraph2DContentLoss,
    dataset={
        "limit": 10,
        "path": GDRIVE_DIR / "MyDrive" / "Thesis" / "Datasets" / "surfaces.zip",
        "transforms": [To(device), View(1, 128, 128)],
    },
)

# + id="3c4cdce1"
training_flow(generator, discriminator)
