import logging
import os
import shutil
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .asha import ASHA
from .backend import Backend, SlurmBackend
from .constants import (
    ASHA_PKL,
    DB_PATH,
    EXPERIMENT_NAME,
    ITERATION,
    STORAGE_PATH,
    TRAIN_PKL,
    TRIAL_ID,
)
from .database import Database
from .storage import Storage
from .suggest import Suggest
from .trial import Status, Trial

_logger = logging.getLogger(__name__)


class Experiment:
    """Run an HPO experiment using random search and the Asynchronous Successive Halving Algorithm (ASHA).

    Args:
        train: A train function that takes as input a `cfg` dict.
        cfg: A dict passed on to the `train` function.
            It must contain the search spaces via `slurm_sweeps.Uniform`, `slurm_sweeps.Choice`, etc.
        name: The name of the experiment.
        local_dir: Where to store and run the experiments. In this directory
            we will create a folder with the experiment name.
        backend: A backend to execute the trials. By default, we choose the `SlurmBackend` if Slurm is available,
            otherwise we choose the standard `Backend` that simply executes the trial in another process.
        asha: An optional ASHA instance to cancel less promising trials. By default, it is None.
        database: A database instance to store the trial's (intermediate) results.
            By default, it will create the database at `{local_dir}/.database'.
        restore: Restore an experiment with the same name?
        exist_ok: Replace an existing experiment with the same name?
    """

    def __init__(
        self,
        train: Callable,
        cfg: Dict,
        name: str = "MySweep",
        local_dir: str = "./slurm_sweeps",
        backend: Optional[Backend] = None,
        asha: Optional[ASHA] = None,
        database: Optional[Database] = None,
        restore: bool = False,
        exist_ok: bool = False,
    ):
        self._cfg = cfg
        self._name = name

        storage_path = self._create_experiment_dir(
            Path(local_dir) / name, restore, exist_ok
        )
        self._storage = Storage(storage_path)
        self._storage.dump(train, TRAIN_PKL)
        if asha:
            self._storage.dump(asha, ASHA_PKL)

        self._database = database or Database(Path(local_dir) / ".database")
        if not restore:
            self._database.create(experiment=self._name, exist_ok=exist_ok)

        self._backend = backend or (
            SlurmBackend() if SlurmBackend.is_available() else Backend()
        )

        self._start_time: Optional[float] = None

        # setting env variables for the logger in the trials
        os.environ[EXPERIMENT_NAME] = name
        os.environ[STORAGE_PATH] = str(self._storage.path)
        os.environ[DB_PATH] = str(self._database.path)

    @staticmethod
    def _create_experiment_dir(
        experiment_path: Path, restore: bool, exist_ok: bool
    ) -> Path:
        experiment_path.mkdir(parents=True, exist_ok=exist_ok)
        if not restore:
            for content in experiment_path.iterdir():
                if content.is_dir():
                    shutil.rmtree(content)
                else:
                    content.unlink()

        return experiment_path

    def run(
        self,
        n_trials: int = 1,
        max_concurrent_trials: Optional[int] = None,
        summary_interval_in_sec: float = 5.0,
        nr_of_rows_in_summary: int = 10,
        summarize_cfg_and_metrics: Union[bool, List[str]] = True,
    ) -> pd.DataFrame:
        """Run the experiment.

        Args:
            n_trials: Number of trials to run.
            max_concurrent_trials: The maximum number of trials running concurrently. By default, we will set this to
                the number of cpus available, or the number of total Slurm tasks divided by the number of trial Slurm
                tasks requested.
            summary_interval_in_sec: Print a summary of the experiment every x seconds.
            nr_of_rows_in_summary: How many rows of the summary table should we print?
            summarize_cfg_and_metrics: Should we include the cfg and the metrics in the summary table?
                You can also pass in a list of strings to only select a few cfg and metric keys.

        Returns:
            A DataFrame of the database.
        """
        max_concurrent_trials = (
            max_concurrent_trials or self._backend.max_concurrent_trials
        )

        _logger.info(
            dedent(
                f"""\
                Running the experiment '{self._name}' ({datetime.today().ctime()})
                    - total number of trials: {n_trials}
                    - max number of concurrent trials: {max_concurrent_trials}
                """
            )
        )

        self._start_time = time.time()
        time_of_last_summary = time.time()

        trials = [Trial(cfg=self._parse(self._cfg)) for _ in range(n_trials)]
        scheduled_trials, running_trials, terminated_trials = copy(trials), [], []

        while len(terminated_trials) < n_trials:
            trial_nr = len(terminated_trials) + len(running_trials)

            # run trial
            if (trial_nr < n_trials) and (len(running_trials) < max_concurrent_trials):
                trial = trials[trial_nr]

                _logger.debug(
                    f"{trial_nr + 1}/{n_trials} run trial {trial.trial_id} with config:\n\t{trial.cfg}"
                )

                trial.process = self._backend.run(trial, self._storage)
                trial.start_time = time.time()

                running_trials.append(trial)
                scheduled_trials.remove(trial)

                continue

            # check for completed trials
            for trial in copy(running_trials):
                if trial.terminated:
                    trial.end_time = time.time()
                    _logger.debug(f"trial {trial.trial_id} {trial.status.value}")

                    terminated_trials.append(trial)
                    running_trials.remove(trial)

            # print summary
            if (time.time() - time_of_last_summary) > summary_interval_in_sec:
                self._print_summary(
                    running_trials + scheduled_trials + terminated_trials,
                    n_rows=nr_of_rows_in_summary,
                    cfg_and_metrics_to_include=None
                    if summarize_cfg_and_metrics is True
                    else (summarize_cfg_and_metrics or []),
                )
                time_of_last_summary = time.time()

            time.sleep(0.1)

        # final summary
        self._print_summary(
            terminated_trials,
            n_rows=None,
            cfg_and_metrics_to_include=None
            if summarize_cfg_and_metrics is True
            else (summarize_cfg_and_metrics or []),
            sort_by="RUNTIME",
        )

        return self._database.read(experiment=self._name)

    def _parse(self, cfg: Dict[str, Any]) -> Dict:
        """Parse the cfg dict applying the search spaces.

        Args:
            cfg: The cfg dict with defined search spaces.

        Returns:
            A cfg dict sampled from the search space.
        """
        suggested_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, Suggest):
                suggested_cfg[key] = val()
            elif isinstance(val, dict):
                suggested_cfg[key] = self._parse(val)
            else:
                suggested_cfg[key] = val

        return suggested_cfg

    def _print_summary(
        self,
        trials: List[Trial],
        n_rows: Optional[int] = None,
        cfg_and_metrics_to_include: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
    ):
        _logger.info("\n=== Status ===")
        elapsed_time = time.time() - self._start_time
        nr_of_ct, nr_of_pt, nr_of_rt, nr_of_st = 0, 0, 0, 0
        for trial in trials:
            if trial.status == Status.COMPLETED:
                nr_of_ct += 1
            elif trial.status == Status.PRUNED:
                nr_of_pt += 1
            elif trial.status == Status.RUNNING:
                nr_of_rt += 1
            elif trial.status == Status.SCHEDULED:
                nr_of_st += 1

        _logger.info(
            f"Elapsed time: {elapsed_time:.0f} s\n"
            f"Terminated trials: {nr_of_ct + nr_of_pt}/{len(trials)} "
            f"({nr_of_rt} running, {nr_of_ct} completed, {nr_of_pt} pruned)"
        )
        _logger.info("---")

        summary_dicts = [
            {
                "TRIAL_ID": trial.trial_id,
                "START_TIME": time.ctime(trial.start_time)
                if trial.start_time
                else None,
                "STATUS": trial.status.value,
                "RUNTIME": trial.runtime,
            }
            for trial in trials
        ]

        # add database info
        database = self._database.read(experiment=self._name)
        if not database.empty:
            if cfg_and_metrics_to_include is None:
                cfg_and_metrics_to_include = [
                    col for col in database.columns if col not in [ITERATION, TRIAL_ID]
                ]

            for trial_dict in summary_dicts:
                id_mask = database[TRIAL_ID] == trial_dict["TRIAL_ID"]
                trial_df = database[id_mask].sort_values(ITERATION)
                if trial_df.empty:
                    continue

                trial_dict["ITERATION"] = trial_df.iloc[-1][ITERATION]
                for key in cfg_and_metrics_to_include:
                    trial_dict[key] = trial_df.iloc[-1][key]

        summary_df = pd.DataFrame(summary_dicts).set_index("TRIAL_ID")
        if sort_by is not None:
            summary_df = summary_df.sort_values(sort_by)
        _logger.info(summary_df.head(n_rows).to_string())
        if n_rows is not None and n_rows < len(summary_df):
            _logger.info(f"... {len(summary_df) - n_rows} trials not displayed!")
