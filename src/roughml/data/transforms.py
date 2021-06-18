from abc import ABC, abstractmethod

from torch import flatten


class Transform(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Flatten(Transform):
    def __call__(self, tensor):
        return flatten(tensor)


class To(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)


class Normalize(Transform):
    def callback(self, dataset):
        self.min = torch.min(dataset.surfaces).item()
        self.max = torch.max(dataset.surfaces).item()

    def __call__(self, tensor):
        if self.max - self.min > 0:
            return (tensor - self.min) / (self.max - self.min)

        return torch.zeros(tensor.size())


class View(Transform):
    def __init__(self, *args):
        self.args = args

    def __call__(self, tensor):
        return tensor.view(*self.args)


# class Pad(Transform):
#     #FIXME
#   def __call__(self, tensor):
#     return np.apply_along_axis(lambda row: np.tile(row, 4), 1, y)
