from typing import TYPE_CHECKING, List

import numpy as np

from .constants import DB_ITERATION, DB_METRIC, DB_TRIAL_ID

if TYPE_CHECKING:
    import pandas as pd


class ASHA:
    """Basic implementation of the Asynchronous Successive Halving Algorithm (ASHA) to prune unpromising trials.

    Args:
        metric: The metric you want to optimize.
        mode: Should the metric be minimized or maximized? Allowed values: ["min", "max"]
        reduction_factor: The reduction factor of the algorithm
        min_t: Minimum number of iterations before we consider pruning.
        max_t: Maximum number of iterations.
    """

    def __init__(
        self,
        metric: str,
        mode: str,
        reduction_factor: int = 4,
        min_t: int = 1,
        max_t: int = 50,
    ):
        self._metric = metric
        self._mode = mode
        self._rf = reduction_factor
        self._min_t = min_t
        self._max_t = max_t

        assert mode == "min" or mode == "max"
        assert reduction_factor > 1
        assert max_t > min_t > 0

        rung_max = int(np.log(self._max_t / self._min_t) / np.log(self._rf))
        self._rungs = [
            self._min_t * (self._rf**i) for i in reversed(range(rung_max + 1))
        ]

    @property
    def metric(self) -> str:
        """The metric to optimize."""
        return self._metric

    @property
    def mode(self) -> str:
        """The 'mode' of the metric, either 'max' or 'min'."""
        return self._mode

    def find_trials_to_prune(self, database: "pd.DataFrame") -> List[str]:
        """Check the database and find trials to prune.

        Args:
            database: The experiment's metrics table of the database as a pandas DataFrame.

        Returns:
            List of trial ids that should be pruned.
        """
        # make sure iteration starts with 1
        if database[DB_ITERATION].min() == 0:
            database[DB_ITERATION] += 1

        metric_column = f"{DB_METRIC}{self._metric}"

        trials = []
        for rung in self._rungs:
            df_r = database[database[DB_ITERATION] == rung]
            if df_r.empty:
                continue

            nans = df_r[metric_column].isna()

            if self._mode == "min":
                cutoff = np.nanpercentile(df_r[metric_column], 1 / self._rf * 100)
                ids = df_r[metric_column] > cutoff
            else:
                cutoff = np.nanpercentile(df_r[metric_column], (1 - 1 / self._rf) * 100)
                ids = df_r[metric_column] < cutoff

            trials += list(df_r[nans | ids][DB_TRIAL_ID])

        return trials
