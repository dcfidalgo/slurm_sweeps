import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .asha import ASHA
from .constants import (
    ASHA_PKL,
    CFG_YML,
    DB_CFG,
    DB_ITERATION,
    DB_LOGGED,
    DB_PATH,
    DB_TRIAL_ID,
    EXPERIMENT_NAME,
    STORAGE_PATH,
    TRIAL_ID,
)
from .database import SqlDatabase
from .storage import Storage


class Logger:
    """Log metrics to a slurm sweeps database and cancel trial if ASHA says so."""

    def __init__(self):
        self._experiment_name = os.environ[EXPERIMENT_NAME]
        self._trial_id = os.environ[TRIAL_ID]
        if not Path(os.environ[DB_PATH]).is_file():
            raise FileNotFoundError(f"Did not find a database at {os.environ[DB_PATH]}")
        self._database = SqlDatabase(os.environ[DB_PATH])

        storage = Storage(os.environ[STORAGE_PATH])
        self._cfg = storage.load(Path(self._trial_id) / CFG_YML)
        self._asha: Optional[ASHA] = None
        try:
            self._asha = storage.load(ASHA_PKL)
        except FileNotFoundError:
            pass

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

        self._database.write(
            self._experiment_name,
            {
                DB_TRIAL_ID: self._trial_id,
                DB_ITERATION: iteration,
                DB_CFG: json.dumps(self._cfg),
                **metrics,
                **{f"{key}{DB_LOGGED}": 1 for key in metrics.keys()},
            },
        )

        if self._asha is not None:
            df = self._database.read(self._experiment_name)
            df = df[~df[f"{self._asha.metric}{DB_LOGGED}"].isna()]
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
