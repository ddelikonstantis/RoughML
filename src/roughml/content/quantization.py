from abc import abstractstaticmethod

from sklearn.preprocessing import KBinsDiscretizer

from roughml.shared.configuration import Configuration


class Quantizer(Configuration):
    """The `Quantizer` sub-classes are responsible for quantizing our
    input data consisting of floating point values. These floating point
    values are going to serve as symbols for the `n-gram graph
    representation` and having an infinite amount of symbols will do us
    no good.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractstaticmethod
    def __call__(self, tensor):
        raise NotImplementedError


class KBinsDiscretizerQuantizer(Configuration):
    def __init__(self, surfaces=None, **kwargs):
        if "encode" not in kwargs:
            kwargs["encode"] = "ordinal"

        self.underlying = KBinsDiscretizer(**kwargs)

        self.original_shape = surfaces.shape[1:]

        self.surfaces = self.underlying.fit_transform(
            surfaces.reshape(surfaces.shape[0], -1)
        )
        self.surfaces = self.surfaces.reshape(*surfaces.shape)

    def __call__(self, tensor):
        return self.underlying.transform(tensor.reshape(1, -1)).reshape(
            *self.original_shape
        )

    def __str__(self):
        return str({"underlying": self.underlying, "shape": self.surfaces.shape})


if __name__ == "__main__":
    tensors = torch.rand(10, 4, 4)

    print(tensors)

    quantizer = KBinsDiscretizerQuantizer(tensors)

    print(quantizer)

    tensor = quantizer(tensors[0])

    print(tensor)
