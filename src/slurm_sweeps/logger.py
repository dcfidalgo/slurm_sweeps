import os
from pathlib import Path
from typing import Dict, Optional, Union

from .asha import ASHA
from .constants import DB_ASHA, DB_PATH, EXPERIMENT_NAME, TRIAL_ID
from .database import SqlDatabase


class Logger:
    """Log metrics to a slurm sweeps database and cancel trial if ASHA says so."""

    def __init__(self):
        self._trial_id = os.environ[TRIAL_ID]

        if not Path(os.environ[DB_PATH]).is_file():
            raise FileNotFoundError(f"Did not find a database at {os.environ[DB_PATH]}")
        self._database = SqlDatabase(os.environ[EXPERIMENT_NAME], os.environ[DB_PATH])

        self._asha: Optional[ASHA] = self._database.load(DB_ASHA)

    def log(self, metrics: Dict[str, Union[float, int]], iteration: int):
        """Log metrics to the database.

        If ASHA is configured, this also checks if the trial needs to be pruned.

        Args:
            metrics: A dictionary containing the metrics.
            iteration: Iteration of the metrics. Most of the time this will be the epoch.

        Raises:
            TrialPruned if the holy ASHA says so!
            ValueError if a metric is not of type `float` or `int`.
        """
        for metric, val in metrics.items():
            if type(val) not in (float, int):
                raise ValueError(
                    f"You can only log metrics of type `float` or `int`. "
                    f"Your metric '{metric}' has type `{type(val)}`."
                )

        self._database.write_metrics(
            trial_id=self._trial_id, iteration=iteration, metrics=metrics
        )

        if self._asha is not None:
            df = self._database.read_metrics(self._asha.metric)
            if self._trial_id in self._asha.find_trials_to_prune(df):
                raise TrialPruned


_LOGGER: Optional[Logger] = None


def log(metrics: Dict[str, Union[float, int]], iteration: int):
    """Log metrics to the database.

    If ASHA is configured, this also checks if the trial needs to be pruned.

    Args:
        metrics: A dictionary containing the metrics.
        iteration: Iteration of the metrics. Most of the time this will be the epoch.

    Raises:
        TrialPruned if the holy ASHA says so!
        ValueError if a metric is not of type `float` or `int`.
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger()

    return _LOGGER.log(metrics, iteration)


class TrialPruned(Exception):
    pass
