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

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/billsioros/thesis/blob/master/Nanorough_surface_Super_resolution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="_XIigYPTAMAH"
# # ✔️ Prerequisites

# + [markdown] id="sv3lJtoU-aoQ"
# First of all we need to take care of a few **prerequisites**, most notably:
#
# - Install the various pip modules that we will be using.
# - Install some linux specific dependencies of our [content loss](#content-loss).
# - Initialize the Random Number Generator(s), so that our experiments can be replicated.
# - Determine:
#   - The current working directory, as it's going to be used to reference various files such as the dataset, our model checkpoints e.t.c
#   - The available hardware backend. GPU utilization is preferable, as it results in higher complition time.
# - `(Optionally)` Mount Google Drive, where we can load our dataset from.

# + [markdown] id="ehWvZSwBXSXI"
# ## Installing [graphviz](https://graphviz.org/) & [libgraphviz-dev](https://packages.debian.org/jessie/libgraphviz-dev)

# + [markdown] id="12JZQTC3YaxN"
# The aforementioned packages are required by [PyINSECT](https://github.com/billsioros/PyINSECT/tree/implementing-HPGs) and more specifically its graph plotting methods.

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="ANi0d4-f57Qc" outputId="908da9e3-f467-4c34-815e-29b23b68b8c8"
# !sudo apt-get install graphviz libgraphviz-dev

# + [markdown] id="f1mdRrYGM60R"
# ## Installing the required `pip` modules

# + [markdown] id="l5FLZ3mv_ho8"
# - [torch](https://pytorch.org/) is our machine learning framework of choice.
# - [numpy](https://numpy.org/), [sympy](https://www.sympy.org/en/index.html) and [scipy](https://www.scipy.org/) are used to in the context of nanorough surface generation.
# - [plotly](https://plotly.com/) (which requires [pandas](https://pandas.pydata.org/)) as well as [matplotlib](https://matplotlib.org/) are used in order to plot various graphs.

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="5fDiW2N-M60S" outputId="db16de1e-3558-495d-f0a0-0d672f11eb37"
# !pip install torch numpy sympy scipy plotly pandas sklearn matplotlib==3.1.1 git+https://github.com/billsioros/PyINSECT.git@FEATURE_Implementing_HPGraphCollector git+https://github.com/billsioros/thesis.git

# + [markdown] id="WktNsUHMAS63"
# ## Initializing (a.k.a `Seeding`) the Random Number Generator(s)

# + [markdown] id="-1mLAArpLX7A"
# We are required to seed various random number generation engines, so that our experiments can be replicated on a later date.

# + cellView="code" id="iw9sr-9MA-dI"
SEED = 1234

import os
import random

import numpy as np

# + cellView="code" id="QMnousxlAbPV"
import torch

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)

# + [markdown] id="zxhexKDEM60T"
# ## Determining the Current Working Directory

# + cellView="code" id="1oRdJanBM60T"
from pathlib import Path

BASE_DIR = Path.cwd()

# + [markdown] id="ptZs2vpwNwJg"
# ## Mounting Google Drive

# + cellView="code" id="NXUKAqjuOvpK"
GDRIVE_DIR = BASE_DIR / "drive"

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="musKsMGXNyQM" outputId="f98fd15d-99ee-4beb-e25a-82069ff8bf49"
try:
    from google.colab import drive

    drive.mount(f"{GDRIVE_DIR}")
except ImportError:
    pass

# + [markdown] id="JflfjrdmDqL0"
# ## Determining available backend

# + [markdown] id="x1IGaiyqYIoI"
# By default, we are going to be utilizing the available CPU backend, if no GPU is available.

# + cellView="code" id="ykoV8K_vDurq"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

# + cellView="code" id="NYf2R4B1VgvI"
device = torch.device(device)

# + [markdown] id="oQ2HamjCaPFw"
# ## Configuring our Loggers

# + cellView="code" id="FDPrOnIyaVQV"
import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s", level=logging.CRITICAL
)

# + id="KXIF2OMgV1rj"
logger = logging.getLogger()
# -

# # 🙃 A naive-approach

# + [markdown] id="n8K2Yt4Yygnp"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="EfhGL9wz_igq"
from roughml.models import PerceptronGenerator

generator = PerceptronGenerator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="oqW-HWLlqW0j" outputId="617dc2ad-e900-4f46-e0a1-ec1c7e9b6f58"
generator

# + cellView="code" id="SllxLU-zqqfO"
from roughml.models import PerceptronDiscriminator

discriminator = PerceptronDiscriminator.from_generator(generator, device=device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="XSirWsR3qZVq" outputId="b981661f-2537-4e03-9901-675c6019f16e"
discriminator

# + [markdown] id="_Dev3j37tn3x"
# ## Training

# + cellView="code" id="nhk6EKKvYcSo"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

# + cellView="code" id="4O4QebF6US_N"
from pathlib import Path

CHECKPOINT_DIR = BASE_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from roughml.content.loss import NGramGraphContentLoss
from roughml.data.transforms import Flatten, To

# + cellView="code" id="ky0tb0YFJQoG"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="iC8dJd352UCq" outputId="c45e8835-d52f-446d-bfc1-eca83af9e53b"
training_flow(generator, discriminator)
# -

# # 😎 A CNN based approach

# + [markdown] id="pz3FhXtfKfVS"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="FfZnXKRwKfVT"
from src.roughml.models import CNNGenerator

generator = CNNGenerator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="wCibc_0iqc3j" outputId="b05a34d2-4cbf-436f-c2f7-9ff31e19ebf7"
generator

# + cellView="code" id="_S2URr5kKfVT"
from src.roughml.models import CNNDiscriminator

discriminator = CNNDiscriminator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="A3zp9MZpqdem" outputId="fca9f23d-8b6d-46c7-db40-ec3baf785806"
discriminator

# + [markdown] id="oUyCo282KWod"
# ## Training

# + cellView="code" id="Pp0NBoc4KWoe"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

from roughml.content.loss import ArrayGraph2DContentLoss
from roughml.data.transforms import To, View

# + cellView="code" id="UrJuNU0tKWoe"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="1m2idVrat7_M" outputId="e698e816-34e0-4ad1-a1f7-1f61a9a04dcf"
training_flow(generator, discriminator)
