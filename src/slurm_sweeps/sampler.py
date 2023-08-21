from typing import Any, List, Optional

import numpy as np


class Random:
    """Abstract class for all random search spaces."""

    def __call__(self) -> Any:
        raise NotImplementedError


class Uniform(Random):
    """Sample from a uniform distribution.

    Args:
        low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
        high: Upper boundary of the output interval. All values generated will be less than high.
        seed: A seed to initialize the RNG. If None, then fresh, unpredictable entropy will be pulled from the OS.
    """

    def __init__(self, low: float, high: float, seed: Optional[int] = None):
        self._low, self._high = low, high
        self._rng = np.random.default_rng(seed=seed)

    def __call__(self) -> float:
        return float(self._rng.uniform(self._low, self._high))


class LogUniform(Random):
    """Sample from a log-uniform distribution.

    Args:
        low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
        high: Upper boundary of the output interval. All values generated will be less than high.
        seed: A seed to initialize the RNG. If None, then fresh, unpredictable entropy will be pulled from the OS.
    """

    def __init__(self, low: float, high: float, seed: Optional[int] = None):
        assert 0 < low < high
        self._low, self._high = np.log(low), np.log(high)
        self._rng = np.random.default_rng(seed=seed)

    def __call__(self) -> float:
        return float(np.exp(self._rng.uniform(self._low, self._high)))


class Choice(Random):
    def __init__(self, choices: List[Any], seed: Optional[int] = None):
        self._choices = choices
        self._type = type(choices[0])
        assert all([isinstance(el, self._type) for el in self._choices])

        self._rng = np.random.default_rng(seed)

    def __call__(self) -> Any:
        return self._type(self._rng.choice(self._choices))
