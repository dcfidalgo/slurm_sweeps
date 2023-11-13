from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from .constants import DB_ITERATION, DB_TRIAL_ID

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
    def metric(self):
        """The metric to optimize."""
        return self._metric

    def find_trials_to_prune(self, database: "pd.DataFrame") -> List[str]:
        """Check the database and find trials to prune.

        Args:
            database: The database of the experiment as a pandas dataframe.

        Returns:
            List of trial ids that should be pruned.
        """
        # make sure iteration starts with 1
        if database[DB_ITERATION].min() == 0:
            database[DB_ITERATION] += 1

        trials = []
        for rung in self._rungs:
            df = database[database[DB_ITERATION] == rung]
            if df.empty:
                continue

            nans = df[self._metric].isna()

            if self._mode == "min":
                cutoff = np.nanpercentile(df[self._metric], 1 / self._rf * 100)
                ids = df[self._metric] > cutoff
            else:
                cutoff = np.nanpercentile(df[self._metric], (1 - 1 / self._rf) * 100)
                ids = df[self._metric] < cutoff

            trials += list(df[nans | ids][DB_TRIAL_ID])

        return trials
