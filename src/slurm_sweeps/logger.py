import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .asha import ASHA
from .constants import (
    ASHA_PKL,
    CFG,
    DB_PATH,
    EXPERIMENT_NAME,
    ITERATION,
    LOGGED,
    STORAGE_PATH,
    TRIAL_ID,
)
from .database import FileDatabase, SqlDatabase
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
        if Path(os.environ[DB_PATH]).is_dir():
            self._database = FileDatabase(os.environ[DB_PATH])
        elif Path(os.environ[DB_PATH]).is_file():
            self._database = SqlDatabase(os.environ[DB_PATH])
        else:
            raise FileNotFoundError(f"Did not find a database at {os.environ[DB_PATH]}")

        self._asha: Optional[ASHA] = None
        try:
            storage = Storage(os.environ[STORAGE_PATH])
            self._asha = storage.load(ASHA_PKL)
        except (KeyError, FileNotFoundError):
            pass

    @property
    def trial(self) -> Trial:
        return self._trial

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
                TRIAL_ID: self._trial.trial_id,
                ITERATION: iteration,
                CFG: json.dumps(self.trial.cfg),
                **metrics,
                **{f"{key}{LOGGED}": 1 for key in metrics.keys()},
            },
        )

        if self._asha is not None:
            df = self._database.read(self._experiment_name)
            df = df[~df[f"{self._asha.metric}{LOGGED}"].isna()]
            if self._trial.trial_id in self._asha.find_trials_to_prune(df):
                raise TrialPruned


class TrialPruned(Exception):
    pass
