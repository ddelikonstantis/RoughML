import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Base(nn.Module):
    @classmethod
    def from_device(
        cls, device, *args, dtype=torch.float64, gradient_clipping=None, **kwargs
    ):
        model = cls(*args, **kwargs)

        # Handle multi-gpu if desired
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, list(range(torch.cuda.device_count())))

        model = model.to(device)

        if dtype is not None:
            model = model.to(dtype=dtype)

        if gradient_clipping is not None:
            if isinstance(gradient_clipping, tuple):
                low, high = gradient_clipping
            else:
                low, high = -gradient_clipping, +gradient_clipping

            for param in model.parameters():
                if param.requires_grad is True:
                    param.register_hook(lambda grad: torch.clamp(grad, low, high))

        return model

    @classmethod
    def from_pt(cls, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        instance = cls()

        load_checkpoint = torch.load(path, map_location="cpu")  # Load to normal memory to avoid GPU memory use here      
        instance.load_state_dict(load_checkpoint)

        instance.eval()

        # Convert model to CUDA version, based on appropriate device
        if device.type != "cpu":
            instance.cuda()

        return instance

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
