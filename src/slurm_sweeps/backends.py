import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional

import yaml

from .constants import CFG_YML, DB_TRAIN, TRAIN_PY
from .database import Database
from .trial import Trial


@dataclass
class SlurmCfg:
    """A configuration class for the SlurmBackend.

    Args:
        exclusive: Add the `--exclusive` switch.
        nodes: How many nodes do you request for your srun?
        ntasks: How many tasks do you request for your srun?
        args: Additional command line arguments for srun, formatted as a string.
    """

    exclusive: bool = True
    nodes: int = 1
    ntasks: int = 1
    args: str = ""


class Backend:
    """The Backend is responsible for running the trials.

    Args:
        execution_dir: From where should I execute the trials?
        database: The database of the experiment.
    """

    def __init__(self, execution_dir: Path, database: Database):
        self._execution_dir = execution_dir.resolve()
        self._database = database

    @property
    def max_concurrent_trials(self):
        """This equals the number of cpus on your machine."""
        return multiprocessing.cpu_count()

    def run(self, trial: Trial) -> subprocess.Popen:
        """Run a trial in a subprocess."""
        trial_dir = self._execution_dir / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        with (trial_dir / TRAIN_PY).open("w") as train_file:
            train_file.write(self._train_script(trial.trial_id))

        with (trial_dir / CFG_YML).open("w") as cfg_file:
            yaml.safe_dump(trial.cfg, cfg_file)

        args = self._build_args(trial_dir / TRAIN_PY)

        out_path = trial_dir / "stdout.log"
        err_path = trial_dir / "stderr.log"
        with out_path.open("w") as out_file, err_path.open("w") as err_file:
            trial_process = subprocess.Popen(
                args,
                shell=True,
                executable="/bin/bash",
                cwd=trial_dir,
                stderr=err_file,
                stdout=out_file,
            )

        return trial_process

    def _train_script(self, trial_id: str) -> str:
        """The python script that executes the train function."""
        template = dedent(
            f"""\
            from slurm_sweeps.database import Database
            from slurm_sweeps.logger import Logger

            database = Database("{self._database.experiment}", "{self._database.path}")
            train = database.load("{DB_TRAIN}")
            trial = database.read_trials(trial_id="{trial_id}")[0]

            Logger(trial_id=trial.trial_id, database=database)
            train(trial.cfg)
            """
        )

        return template

    @staticmethod
    def _build_args(train_path: Path) -> str:
        """Build arguments for the subprocess."""
        return f"python {train_path}"


class SlurmBackend(Backend):
    """Execute the training runs on a Slurm cluster via `srun`.

    Pass an instance of this class to your experiment.

    Args:
        execution_dir: From where should I execute the trials?
        database: The database of the experiment.
        cfg: A `SlurmCfg` instance.
    """

    def __init__(
        self, execution_dir: Path, database: Database, cfg: Optional[SlurmCfg] = None
    ):
        super().__init__(execution_dir=execution_dir, database=database)

        self._cfg = cfg or SlurmCfg()

    @property
    def max_concurrent_trials(self) -> int:
        """This equals the total number of SLURM tasks requested."""
        return int(int(os.environ["SLURM_NTASKS"]) / self._cfg.ntasks)

    def _build_args(self, train_path: Path) -> str:
        """Build arguments for the subprocess."""
        slurm_cmd = (
            f"srun {'--exclusive' if self._cfg.exclusive else ''} "
            f"--nodes={self._cfg.nodes} --ntasks={self._cfg.ntasks} {self._cfg.args} "
        )
        return slurm_cmd + super()._build_args(train_path)

    @staticmethod
    def is_running() -> bool:
        """Are we running inside an `sbatch` call?"""
        try:
            subprocess.check_output(["sinfo"])
            assert "SLURM_NTASKS" in os.environ
        except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
            return False
        else:
            return True
