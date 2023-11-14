import json
import logging
import os
import shutil
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from .asha import ASHA
from .backend import Backend, SlurmBackend
from .constants import (
    ASHA_PKL,
    DB_CFG,
    DB_ITERATION,
    DB_LOGGED,
    DB_PATH,
    DB_TIMESTAMP,
    DB_TRIAL_ID,
    EXPERIMENT_NAME,
    STORAGE_PATH,
    TRAIN_PKL,
    TRIAL_ID,
    WAITING_TIME_IN_SEC,
)
from .database import ExperimentExistsError, SqlDatabase
from .sampler import Sampler
from .storage import Storage
from .trial import Status, Trial

_logger = logging.getLogger(__name__)


class Experiment:
    """Set up an HPO experiment.

    Args:
        train: A train function that takes as input the `cfg` dict.
        cfg: A dict passed on to the `train` function.
            It must contain the search spaces via `slurm_sweeps.Uniform`, `slurm_sweeps.Choice`, etc.
        name: The name of the experiment.
        local_dir: Where to store and run the experiments. In this directory
            we will create the database `slurm_sweeps.db` and a folder with the experiment name.
        backend: A backend to execute the trials. By default, we choose the `SlurmBackend` if Slurm is available,
            otherwise we choose the standard `Backend` that simply executes the trial in another process.
        asha: An optional ASHA instance to cancel less promising trials.
        restore: Restore an experiment with the same name?
        overwrite: Overwrite an existing experiment with the same name?
    """

    def __init__(
        self,
        train: Callable,
        cfg: Dict,
        name: str = "MySweep",
        local_dir: Union[str, Path] = "./slurm-sweeps",
        backend: Optional[Backend] = None,
        asha: Optional[ASHA] = None,
        restore: bool = False,
        overwrite: bool = False,
    ):
        self._cfg = cfg
        self._name = name
        self._local_dir = Path(local_dir)

        storage_path = self._create_experiment_dir(
            self._local_dir / name, restore, overwrite
        )
        self._storage = Storage(storage_path)
        self._storage.dump(train, TRAIN_PKL)
        if asha:
            self._storage.dump(asha, ASHA_PKL)

        self._database = SqlDatabase(self._local_dir / "slurm_sweeps.db")
        if not restore:
            self._database.create(experiment=self._name, overwrite=overwrite)

        self._backend = backend or (
            SlurmBackend() if SlurmBackend.is_running() else Backend()
        )

        self._start_time: Optional[float] = None

        # setting env variables for the logger in the trials
        os.environ[EXPERIMENT_NAME] = name
        os.environ[STORAGE_PATH] = str(self._storage.path)
        os.environ[DB_PATH] = str(self._database.path)
        # in case we import modules that are not installed
        try:
            os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.getcwd()}"
        except KeyError:
            os.environ["PYTHONPATH"] = os.getcwd()

    @staticmethod
    def _create_experiment_dir(
        experiment_path: Path, restore: bool, exist_ok: bool
    ) -> Path:
        try:
            experiment_path.mkdir(parents=True, exist_ok=exist_ok)
        except FileExistsError as err:
            raise ExperimentExistsError(experiment_path.name) from err
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
            n_trials: Number of trials to run. For grid searches this parameter is ignored.
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
        trials = [Trial(cfg=cfg) for cfg in Sampler(self._cfg, n_trials)]
        max_concurrent_trials = (
            max_concurrent_trials or self._backend.max_concurrent_trials
        )
        self._print_run_info(len(trials), max_concurrent_trials)

        self._start_time = time.time()
        time_of_last_summary = time.time()

        scheduled_trials, running_trials, terminated_trials = copy(trials), [], []

        while len(terminated_trials) < n_trials:
            trial_nr = len(terminated_trials) + len(running_trials)

            # run trial
            if (trial_nr < n_trials) and (len(running_trials) < max_concurrent_trials):
                trial = self._run_trial(trials, trial_nr)

                running_trials.append(trial)
                scheduled_trials.remove(trial)

                continue

            # check for terminated trials
            for trial in copy(running_trials):
                if trial.terminated:
                    trial.end_time = time.time()

                    _logger.debug(f"trial {trial.trial_id} {trial.status.value}")

                    terminated_trials.append(trial)
                    running_trials.remove(trial)

            # print current summary
            if (time.time() - time_of_last_summary) > summary_interval_in_sec:
                self._print_summary(
                    running_trials + scheduled_trials + terminated_trials,
                    n_rows=nr_of_rows_in_summary,
                    summarize_cfg_and_metrics=summarize_cfg_and_metrics,
                )
                time_of_last_summary = time.time()

            # wait if the maximum nr of concurrent trials are running, or wait for the last trials to finish
            if len(running_trials) == max_concurrent_trials or trial_nr == n_trials:
                time.sleep(WAITING_TIME_IN_SEC)

        # print final summary
        self._print_summary(
            terminated_trials,
            n_rows=None,
            summarize_cfg_and_metrics=summarize_cfg_and_metrics,
            sort_by="RUNTIME",
        )

        return self._database.read(experiment=self._name)

    def _run_trial(self, trials: List[Trial], trial_nr: int) -> Trial:
        trial = trials[trial_nr]

        _logger.debug(
            f"{trial_nr}/{len(trials)}: run trial {trial.trial_id} with config:\n\t{trial.cfg}"
        )

        trial.process = self._backend.run(trial, self._storage)
        trial.start_time = time.time()

        return trial

    def _print_run_info(self, nr_trials: int, max_concurrent_trials: int):
        _logger.info(
            dedent(
                f"""\
                Running the experiment '{self._name}' ({datetime.today().ctime()})
                    - total number of trials: {nr_trials}
                    - max number of concurrent trials: {max_concurrent_trials}
                """
            )
        )

    def _print_summary(
        self,
        trials: List[Trial],
        n_rows: Optional[int] = None,
        summarize_cfg_and_metrics: Union[bool, List[str]] = True,
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
            metrics = [
                col
                for col in database.columns
                if not (col.startswith("_") or col.endswith(DB_LOGGED))
            ]

            if summarize_cfg_and_metrics is True:
                summarize_cfg_and_metrics = list(
                    json.loads(database[DB_CFG].iloc[0]).keys()
                )
                summarize_cfg_and_metrics += metrics
            elif summarize_cfg_and_metrics is False:
                summarize_cfg_and_metrics = []

            for trial_dict in summary_dicts:
                id_mask = database[DB_TRIAL_ID] == trial_dict["TRIAL_ID"]
                trial_df = database[id_mask].sort_values(DB_ITERATION)
                if trial_df.empty:
                    continue

                trial_dict["ITERATION"] = trial_df.iloc[-1][DB_ITERATION]
                # adding cfg
                for key, val in json.loads(trial_df.iloc[-1][DB_CFG]).items():
                    if key in summarize_cfg_and_metrics:
                        trial_dict[key] = val

                # adding metrics
                for metric in metrics:
                    if metric not in summarize_cfg_and_metrics:
                        continue
                    metric_df = trial_df[trial_df[f"{metric}{DB_LOGGED}"] == 1]
                    trial_dict[metric] = metric_df.iloc[-1][metric]

        summary_df = pd.DataFrame(summary_dicts).set_index("TRIAL_ID")

        if sort_by is not None:
            summary_df = summary_df.sort_values(sort_by)

        _logger.info(summary_df.head(n_rows).to_string())

        if n_rows is not None and n_rows < len(summary_df):
            _logger.info(f"... {len(summary_df) - n_rows} trials not displayed!")
