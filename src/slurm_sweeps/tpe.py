import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats.distributions import norm

from .database import Database
from .sampler import Grid, Random, Sampler, Uniform


class NumericalParzenEstimator:
    def __init__(
        self,
        dataset: np.array,
        low: float,
        high: float,
        search_space_dim: int = 1,
        min_bandwidth: float = 0.03,
    ):
        self._dataset = dataset
        self._size = len(dataset) + 1  # adding the prior
        self._low = low
        self._high = high

        SIGMA0_MAGNITUDE = 0.2
        self._bandwidth = (
            SIGMA0_MAGNITUDE
            * max(self._size, 1) ** (-1.0 / (search_space_dim + 4))
            * (high - low)
        )
        self._bandwidth = max(self._bandwidth, min_bandwidth * (high - low))

        self._kernels = [norm(loc=(low + high) / 2, scale=(high - low))]
        self._kernels += [norm(loc=loc, scale=self._bandwidth) for loc in self._dataset]

    @property
    def bandwidth(self):
        return self._bandwidth

    def pdf(self, x: np.array) -> np.array:
        x = np.atleast_1d(x)
        values = sum([kernel.pdf(x) for kernel in self._kernels]) / self._size

        return values

    def sample(self, size: int = 1, rng: np.random._generator.Generator = None):
        rng = rng or np.random.default_rng()
        idxs = rng.choice(self._size, size=size)
        sample = np.array([self._sample(idx, rng) for idx in idxs])

        return sample.squeeze()

    def _sample(self, idx: int, rng: np.random._generator.Generator) -> np.array:
        for _ in range(100):
            value = self._kernels[idx].rvs(size=1, random_state=rng)
            if self._low <= value <= self._high:
                return value

        raise ValueError(
            f"Sampling took too long! Could not sample a value between '{self._low}' and '{self._high}' for "
            f"kernel with loc '{self._kernels[idx].mean()}' and scale '{self._kernels[idx].std()}'"
        )


@dataclass
class TpeConfig:
    """A configuration class for the TPE algorithm."""

    fraction: float = 0.1
    gamma: float = 0.25
    min_bandwidth: float = 0.03
    n_ei: int = 24
    n_i: int = 10


class TPE:
    def __init__(
        self,
        metric: str,
        mode: Literal["min", "max"],
        database: Database,
        config: TpeConfig,
    ):
        self._metric = metric
        self._mode = mode
        self._database = database
        self._config = config

        assert self._mode == "min" or self._mode == "max"

        self._data: pd.DataFrame = None
        self._rung: Optional[int] = None

    def update_data(self):
        """Get updated data from the database and set the rung."""
        data_points = self._database.read_data_for_tpe(self._metric)
        if not data_points:
            return

        df = pd.DataFrame(
            [{"_iteration": dp.iteration, "_metric": dp.metric} for dp in data_points]
        )
        flattened_cfg = pd.json_normalize([dp.cfg for dp in data_points])

        self._data = pd.concat([df, flattened_cfg], axis=1)

        # Iterate through the rungs in descending order and check if we have enough seen values.
        # Else, self._rung is None and `self.__call__` will perform a random sampling.
        self._rung = None
        sorted_df = self._data.sort_values("_iteration", ascending=False)
        for group in sorted_df.groupby("_iteration", sort=False):
            if group[1]["_metric"].count() >= self._config.n_i:
                self._rung = group[0]
                break

    def __call__(self, parameter: str, search_space: Uniform) -> float:
        if self._data is None or self._rung is None:
            return search_space()

        rung_mask = self._data["_iteration"] == self._rung
        seen_values = self._data[parameter][rung_mask].values
        best_mask = self._compute_mask(self._data["_metric"][rung_mask])

        return self._tpe(seen_values, best_mask, search_space.low, search_space.high)

    def _compute_mask(self, metrics: np.ndarray) -> np.ndarray:
        """Compute the mask that corresponds to the best metric values.

        Args:
            metrics:
        """
        best = np.zeros_like(metrics).astype(bool)
        idx_sorted = np.argsort(metrics)
        if self._mode == "max":
            idx_sorted = idx_sorted[::-1]

        n = int(np.round(self._config.fraction * len(metrics)))
        best[idx_sorted[:n]] = True

        return best

    def _tpe(self, xs: np.ndarray, mask: np.ndarray, low: float, high: float) -> float:
        """Suggest a new parameter value for the next trial.

        Args:
            xs: Already seen values.
            mask: Mask for the seen values that corresponds to the best metric values.

        Returns:
            Suggested value for the next trial.
        """
        k_b = NumericalParzenEstimator(
            xs[mask], low=low, high=high, min_bandwidth=self._config.min_bandwidth
        )
        k_w = NumericalParzenEstimator(
            xs[~mask], low=low, high=high, min_bandwidth=self._config.min_bandwidth
        )

        candidates = k_b.sample(self._config.n_ei)
        best_value = candidates[np.argmax(k_b.pdf(candidates) / k_w.pdf(candidates))]

        return float(best_value)


class TpeSampler(Sampler):
    def __init__(self, cfg: Dict[str, Any], tpe: TPE):
        super().__init__(cfg=cfg)

        self._tpe = tpe

    def __call__(self):
        self._tpe.update_data()

        return super().__call__()

    def _sample(
        self,
        cfg: Dict[str, Any],
        grid_values: Optional[List[Any]] = None,
        parent: str = "",
    ):
        suggested_cfg = {}
        for key, val in cfg.items():
            parameter = f"{parent}.{key}" if parent else key
            if isinstance(val, Uniform):
                # Only apply TPE for the linear continuous search space for now
                suggested_cfg[key] = self._tpe(parameter=parameter, search_space=val)
            elif isinstance(val, Random):
                warnings.warn(
                    "For now, we only apply the TPE to the linear continuous search space `Uniform`!"
                )
                suggested_cfg[key] = val()
            elif isinstance(val, Grid):
                suggested_cfg[key] = grid_values.pop(0)
            elif isinstance(val, dict):
                suggested_cfg[key] = self._sample(val, grid_values, parent=parameter)
            else:
                suggested_cfg[key] = val

        return suggested_cfg
