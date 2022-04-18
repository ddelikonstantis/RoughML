import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

from roughml.models.base import Base


# Creating a DeepAutoencoder class
class DeepAutoencoder(Base):
    def __init__(self, channels=1024):
        super().__init__()

        self.channels = channels

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, channels),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded