import itertools
from typing import Any, Dict, List, Optional

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
        self.low, self.high = low, high
        self._rng = np.random.default_rng(seed=seed)

    def __call__(self) -> float:
        return float(self._rng.uniform(self.low, self.high))


class LogUniform(Random):
    """Sample from a log-uniform distribution.

    Args:
        low: Lower boundary of the output interval. All values generated will be greater than or equal to low.
        high: Upper boundary of the output interval. All values generated will be less than high.
        seed: A seed to initialize the RNG. If None, then fresh, unpredictable entropy will be pulled from the OS.
    """

    def __init__(self, low: float, high: float, seed: Optional[int] = None):
        assert 0 < low < high
        self.low, self.high = np.log(low), np.log(high)
        self._rng = np.random.default_rng(seed=seed)

    def __call__(self) -> float:
        return float(np.exp(self._rng.uniform(self.low, self.high)))


class Choice(Random):
    """Sample uniformly from a list of choices.

    Args:
        choices: A list of choices to randomly sample from.
        seed: A seed to initialize the RNG. If None, then fresh, unpredictable entropy will be pulled from the OS.

    Raises:
        AssertionError if the list of choices contains more then one type.
    """

    def __init__(self, choices: List[Any], seed: Optional[int] = None):
        self._choices = choices
        self._type = type(choices[0])
        assert all([isinstance(el, self._type) for el in self._choices])

        self._rng = np.random.default_rng(seed)

    def __call__(self) -> Any:
        return self._type(self._rng.choice(self._choices))


class Grid:
    """Define a grid search space.

    Args:
        values: The grid values
    """

    def __init__(self, values: List[Any]):
        self._values = values

    def __call__(self) -> List[Any]:
        return self._values


class Sampler:
    """A sampler for random and grid search.

    Use its call method to sample from the cfg dict.

    Args:
        cfg: The cfg dict.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg

        self._grid = None
        self._grid_axes = self._extract_grid_axes(cfg)
        if self._grid_axes:
            self._grid = itertools.product(*self._grid_axes)

    def _extract_grid_axes(self, cfg: Dict[str, Any]) -> List[List[Any]]:
        grids = []
        for val in cfg.values():
            if isinstance(val, Grid):
                grids.append(val())
            elif isinstance(val, dict):
                grids += self._extract_grid_axes(val)

        return grids

    def __call__(self) -> dict:
        grid_values = None
        if self._grid:
            try:
                grid_values = list(next(self._grid))
            except StopIteration:
                self._grid = itertools.product(*self._grid_axes)
                grid_values = list(next(self._grid))

        return self._sample(self._cfg, grid_values)

    def _sample(self, cfg: Dict[str, Any], grid_values: Optional[List[Any]] = None):
        suggested_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, Random):
                suggested_cfg[key] = val()
            elif isinstance(val, Grid):
                suggested_cfg[key] = grid_values.pop(0)
            elif isinstance(val, dict):
                suggested_cfg[key] = self._sample(val, grid_values)
            else:
                suggested_cfg[key] = val

        return suggested_cfg
