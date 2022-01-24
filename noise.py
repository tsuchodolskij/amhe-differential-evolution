from abc import abstractmethod
import random

import perlin_noise as pn


class Noise:
    def __init__(self, amplitude=1.0):
        self.amplitude = amplitude

    @abstractmethod
    def _function(self, point) -> float:
        return 0.0

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        pass

    def evaluate(self, point) -> float:
        return self._function(point) * self.amplitude


class NoNoise(Noise):
    def __init__(self):
        super().__init__()

    def _function(self, point) -> float:
        return 0.0

    def set_seed(self, seed: int) -> None:
        pass


class PerlinNoise(Noise):
    def __init__(self, amplitude=1.0):
        super().__init__(amplitude)
        self.noise = pn.PerlinNoise()

    def set_seed(self, seed: int) -> None:
        self.noise = pn.PerlinNoise(seed=seed)

    def _function(self, point) -> float:
        return self.noise(point)


class WhiteNoise(Noise):
    def __init__(self, amplitude=1.0):
        super().__init__(amplitude)
        self.noise = random

    def set_seed(self, seed: int) -> None:
        self.noise.seed(seed)

    def _function(self, point) -> float:
        return self.noise.random()


class GaussianNoise(Noise):
    def __init__(self, amplitude=1.0, sigma=1.0):
        super().__init__(amplitude)
        self.noise = random
        self.sigma = sigma

    def set_seed(self, seed: int) -> None:
        self.noise.seed(seed)

    def _function(self, point) -> float:
        return self.noise.gauss(0.0, self.sigma)
