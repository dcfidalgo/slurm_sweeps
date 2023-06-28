import os
from typing import Dict

from .constants import (
    ASHA_PKL,
    DB_PATH,
    EXPERIMENT_NAME,
    ITERATION,
    STORAGE_PATH,
    TRIAL_ID,
)
from .database import Database
from .storage import Storage
from .trial import Trial


class Logger:
    """Log metrics to a slurm sweeps database and cancel trial if ASHA says so.

    Args:
        cfg: The cfg dict of the train function.
    """

    def __init__(self, cfg: Dict):
        self._trial = Trial(cfg=cfg)

        self._experiment_name = os.environ[EXPERIMENT_NAME]
        self._database = Database(os.environ[DB_PATH])

        try:
            storage = Storage(os.environ[STORAGE_PATH])
            self._asha = storage.load(ASHA_PKL)
        except (KeyError, FileNotFoundError):
            self._asha = None

    @property
    def trial(self) -> Trial:
        return self._trial

    def log(self, key: str, value: float, iteration: int):
        """Log a metric to the database.

        If ASHA is configured, this also checks if the trial needs to be pruned.

        Args:
            key: Name of the metric.
            value: Value of the metric.
            iteration: Iteration of the metric. Most of the time this will be the epoch.

        Raises:
            TrialPruned if the holy ASHA says so!
        """
        row = self._trial.cfg.copy()
        row.update(
            {
                TRIAL_ID: self._trial.trial_id,
                ITERATION: iteration,
            }
        )
        row[key] = value

        self._database.write(self._experiment_name, row)

        if self._asha is not None:
            db = self._database.read(self._experiment_name)
            if self._trial.trial_id in self._asha.find_trials_to_prune(db):
                raise TrialPruned


class TrialPruned(Exception):
    pass
