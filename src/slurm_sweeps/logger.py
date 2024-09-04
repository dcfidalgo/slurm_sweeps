from typing import Dict, Optional, Union

from .asha import ASHA
from .constants import DB_ASHA
from .database import Database


class Logger:
    """Log metrics to a slurm sweeps database and cancel trial if ASHA says so.

    This class is a singleton!

    Args:
        trial_id:
        database:
    """

    instance: Optional["Logger"] = None

    def __init__(self, trial_id: str, database: Database):
        self._trial_id = trial_id
        self._database = database

        self._asha: Optional[ASHA] = self._database.load(DB_ASHA)

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    @property
    def trial_id(self) -> str:
        return self._trial_id

    def log(self, metrics: Dict[str, Union[float, int]], iteration: int):
        """Log metrics to the database.

        If ASHA is configured, this also checks if the trial needs to be pruned.

        Args:
            metrics: A dictionary containing the metrics.
            iteration: Iteration of the metrics. Most of the time this will be the epoch.

        Raises:
            `TrialPruned` if the holy ASHA says so!
            `TypeError` if a metric is not of type `float` or `int`.
        """
        for metric, val in metrics.items():
            if type(val) not in (float, int):
                raise TypeError(
                    f"You can only log metrics of type `float` or `int`. "
                    f"Your metric '{metric}' has type `{type(val)}`."
                )

        self._database.write_metrics(
            trial_id=self.trial_id, iteration=iteration, metrics=metrics
        )

        if self._asha is not None:
            df = self._database.read_metrics(self._asha.metric)
            if self.trial_id in self._asha.find_trials_to_prune(df):
                raise TrialPruned


def log(metrics: Dict[str, Union[float, int]], iteration: int):
    """Log metrics to the database.

    If ASHA is configured, this also checks if the trial needs to be pruned.

    Args:
        metrics: A dictionary containing the metrics.
        iteration: Iteration of the metrics. Most of the time this will be the epoch.

    Raises:
        `TrialPruned` if the holy ASHA says so!
        `TypeError` if a metric is not of type `float` or `int`.
    """
    logger = Logger.instance
    if logger is None:
        raise TypeError(
            "You first have to instantiate the `slurm_sweeps.logger.Logger` singleton "
            "before calling the `slurm_sweeps.logger.log` function."
        )

    return logger.log(metrics, iteration)


class TrialPruned(Exception):
    pass
