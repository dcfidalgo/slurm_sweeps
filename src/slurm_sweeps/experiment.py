import logging
import shutil
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from .asha import ASHA
from .backends import Backend, SlurmBackend, SlurmCfg
from .constants import (
    DB_ASHA,
    DB_CFG,
    DB_ITERATION,
    DB_LOGGED,
    DB_METRIC,
    DB_TRAIN,
    DB_TRIAL_ID,
    WAITING_TIME_IN_SEC,
)
from .database import ExperimentExistsError, ExperimentNotFoundError, SqlDatabase
from .sampler import Sampler
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
        slurm_cfg: A `SlurmCfg` instance passed on to the `slurm_sweeps.backends.SlurmBackend`.
            We automatically choose this backend when slurm sweeps is used within a sbatch script.
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
        asha: Optional[ASHA] = None,
        slurm_cfg: Optional[SlurmCfg] = None,
        restore: bool = False,
        overwrite: bool = False,
    ):
        self._cfg = cfg
        self._name = name
        self._local_dir = Path(local_dir)

        self._create_experiment_dir(self._local_dir / name, restore, overwrite)

        self._database = SqlDatabase(self.name, self.local_dir / "slurm_sweeps.db")
        if not restore:
            self._database.create(overwrite=overwrite)
        elif not self._database.exists():
            raise ExperimentNotFoundError(self.name)

        self._database.dump({DB_TRAIN: train, DB_ASHA: asha})

        if SlurmBackend.is_running():
            self._backend = SlurmBackend(
                execution_dir=self._local_dir / name,
                database=self._database,
                cfg=slurm_cfg,
            )
        else:
            self._backend = Backend(
                execution_dir=self._local_dir / name, database=self._database
            )

        self._start_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    @property
    def local_dir(self) -> Path:
        """The local directory of the experiment."""
        return self._local_dir

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
        trials = [
            Trial(cfg=cfg, status="scheduled") for cfg in Sampler(self._cfg, n_trials)
        ]
        max_concurrent_trials = (
            max_concurrent_trials or self._backend.max_concurrent_trials
        )
        self._print_run_info(len(trials), max_concurrent_trials)

        self._start_time = datetime.now()
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
                if trial.is_terminated():
                    trial.end_time = datetime.now()

                    _logger.debug(f"trial {trial.trial_id} {trial.status.value}")

                    self._database.write_trial(trial)

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
        summary_df = self._print_summary(
            terminated_trials,
            n_rows=None,
            summarize_cfg_and_metrics=summarize_cfg_and_metrics,
            sort_by="RUNTIME",
        )

        return summary_df

    def _run_trial(self, trials: List[Trial], trial_nr: int) -> Trial:
        trial = trials[trial_nr]

        # first write trial to database, then call backend.run !!!
        self._database.write_trial(trial)

        _logger.debug(
            f"{trial_nr}/{len(trials)}: run trial {trial.trial_id} with config:\n\t{trial.cfg}"
        )

        trial.process = self._backend.run(trial)
        trial.start_time = datetime.now()

        # update the status
        self._database.write_trial(trial)

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
    ) -> pd.DataFrame:
        _logger.info("\n=== Status ===")

        elapsed_time = datetime.now() - self._start_time
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
            f"Elapsed time: {elapsed_time.seconds} s\n"
            f"Terminated trials: {nr_of_ct + nr_of_pt}/{len(trials)} "
            f"({nr_of_rt} running, {nr_of_ct} completed, {nr_of_pt} pruned)"
        )
        _logger.info("---")

        metrics_db = self._database.read_metrics()

        cfg_and_metrics_to_summarize = []
        if summarize_cfg_and_metrics is True:
            cfg_and_metrics_to_summarize = list(trials[0].cfg.keys())
            if not metrics_db.empty:
                cfg_and_metrics_to_summarize += [
                    col
                    for col in metrics_db.columns
                    if (col.startswith(DB_METRIC) and not col.endswith(DB_LOGGED))
                ]
        elif isinstance(summarize_cfg_and_metrics, list):
            cfg_and_metrics_to_summarize = summarize_cfg_and_metrics

        summary_dicts = []
        for trial in trials:
            summary_i = {
                "TRIAL_ID": trial.trial_id,
                "START_TIME": trial.start_time.ctime() if trial.start_time else None,
                "STATUS": trial.status.value,
                "RUNTIME": trial.runtime,
            }
            if not metrics_db.empty:
                id_mask = metrics_db[DB_TRIAL_ID] == trial.trial_id
                trial_df = metrics_db[id_mask].sort_values(DB_ITERATION)
                if trial_df.empty:
                    summary_dicts.append(summary_i)
                    continue

                summary_i["ITERATION"] = trial_df.iloc[-1][DB_ITERATION]
                summary_i.update(
                    {
                        k: v
                        for k, v in trial.cfg.items()
                        if k in cfg_and_metrics_to_summarize
                    }
                )

                for metric in cfg_and_metrics_to_summarize:
                    if metric not in trial_df.columns:
                        continue
                    metric_df = trial_df[trial_df[f"{metric}{DB_LOGGED}"] == 1]
                    if not metric_df.empty:
                        summary_i[metric.replace(DB_METRIC, "", 1)] = metric_df.iloc[
                            -1
                        ][metric]
            summary_dicts.append(summary_i)

        summary_df = pd.DataFrame(summary_dicts).set_index("TRIAL_ID")

        if sort_by is not None:
            summary_df = summary_df.sort_values(sort_by)

        _logger.info(summary_df.head(n_rows).to_string())

        if n_rows is not None and n_rows < len(summary_df):
            _logger.info(f"... {len(summary_df) - n_rows} trials not displayed!")

        return summary_df
