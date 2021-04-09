from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import sympy
from scipy import stats


def debug(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        rv = method(*args, **kwargs)

        called_with = ''
        if args:
            called_with += ', '.join(str(x) for x in args)

        if kwargs:
            if args:
                called_with += ', '
            called_with += ', '.join(f"{x}={y}" for x, y in kwargs.items())

        print(f"{method.__name__}({called_with}) returned {rv}")

        return rv

    return wrapper


class SurfaceGenerator(ABC):
    def __init__(self, n_points=500, rms=1, skewness=0, kurtosis=3, corlength_x=20, corlength_y=20, alpha=1):
        self.n_points = n_points
        self.rms = rms
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.corlength_x = corlength_x
        self.corlength_y = corlength_y
        self.alpha = alpha

        self._mean = 0
        self._length = 0

    def __str__(self):
        return f"{self.__class__.__name__}({self.n_points}, {self.rms}, {self.skewness}, {self.kurtosis}, {self.corlength_x}, {self.corlength_y}, {self.alpha})"

    def __repr__(self):
        return f'<{self}>'

    def sort(self, elements):
        indices = np.argsort(elements, axis=0)

        return elements[indices], indices

    @abstractmethod
    def autocorrelation(self, tx, ty):
        raise NotImplementedError

    def generate_surface(self):
        # 1st step: Generation of a Gaussian surface

        # Determine the autocorrelation function R(tx,ty)
        R = np.zeros((self.n_points, self.n_points))

        txmin = -self.n_points // 2
        txmax = self.n_points // 2

        tymin = -self.n_points // 2
        tymax = self.n_points // 2

        dtx = (txmax - txmin) // self.n_points
        dty = (tymax-tymin) // self.n_points

        for tx in range(txmin, txmax, dtx):
            for ty in range(tymin, tymax, dty):
                R[tx + txmax, ty + tymax] = self.autocorrelation(tx, ty)

        # According to the Wiener-Khinchine theorem FR is the power spectrum of the desired profile
        FR = np.fft.fft2(R, (self.n_points, self.n_points))
        AMPR = np.sqrt(dtx ** 2 + dty ** 2) * abs(FR)

        # 2nd step: Generate a white noise, normalize it and take its Fourier transform
        X = np.random.rand(self.n_points, self.n_points)
        aveX = np.mean(np.mean((X)))

        dif2X = (X - aveX) ** 2
        stdX = np.sqrt(np.mean(np.mean(dif2X)))
        X = X / stdX
        XF = np.fft.fft2(X, s=(self.n_points, self.n_points))

        # 3nd step: Multiply the two Fourier transforms
        YF = XF * np.sqrt(AMPR)

        # 4th step: Perform the inverse Fourier transform of YF and get the desired surface
        zaf = np.fft.ifft2(YF, s=(self.n_points, self.n_points))
        z = np.real(zaf)

        avez = np.mean(np.mean(z))
        dif2z = (z-avez) ** 2
        stdz = np.sqrt(np.mean(np.mean(dif2z)))
        z = ((z - avez) * self.rms) / stdz

        # Define the fraction of the surface to be analysed
        xmin = 0
        xmax = self.n_points
        ymin = 0
        ymax = self.n_points
        z_gs = z[xmin:xmax, ymin:ymax]

        # 2nd step: Generation of a non-Gaussian noise NxN
        z_ngn = stats.pearson3.rvs(
            self.skewness,
            loc=self._mean, scale=self.rms, size=(self.n_points, self.n_points)
        )

        # 3rd step: Combination of z_gs with z_ngn to output a z_ms
        v_gs = z_gs.flatten()
        v_ngn = z_ngn.flatten()

        _, Igs = self.sort(v_gs)

        vs_ngn, _ = self.sort(v_ngn)

        v_ngs = vs_ngn[Igs]

        return v_ngs.reshape(self.n_points, self.n_points)

    def __call__(self, length):
        self._length = length

        return self

    def __iter__(self):
        for _ in range(self._length):
            yield self.generate_surface()


class NonGaussianSurfaceGenerator(SurfaceGenerator):
    def __init__(self, n_points=500, rms=1, skewness=0, kurtosis=3, corlength_x=20, corlength_y=20, alpha=1):
        super().__init__(
            n_points=n_points, rms=rms, skewness=skewness, kurtosis=kurtosis,
            corlength_x=corlength_x, corlength_y=corlength_y, alpha=alpha
        )

    def autocorrelation(self, tx, ty):
        return ((self.rms ** 2) * np.exp(-(abs(np.sqrt((tx / self.corlength_x) ** 2 + (ty / self.corlength_y) ** 2))) ** (2 * self.alpha)))


class BeselNonGaussianSurfaceGenerator(NonGaussianSurfaceGenerator):
    def __init__(self, n_points=500, rms=1, skewness=0, kurtosis=3, corlength_x=20, corlength_y=20, alpha=1, beta=1):
        super().__init__(
            n_points=n_points, rms=rms, skewness=skewness, kurtosis=kurtosis,
            corlength_x=corlength_x, corlength_y=corlength_y, alpha=alpha
        )

        self.beta = beta

    def autocorrelation(self, tx, ty):
        return super().autocorrelation(tx, ty) * sympy.besselj(0, (2 * np.pi * np.sqrt(tx ** 2 + ty ** 2)) / self.beta)


if __name__ == '__main__':
    g = BeselNonGaussianSurfaceGenerator()
    # g = NonGaussianSurfaceGenerator()

    for surface in g(1):
        print(surface)
