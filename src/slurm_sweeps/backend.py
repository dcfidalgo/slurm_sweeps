import multiprocessing
import os
import subprocess
from pathlib import Path
from textwrap import dedent

from .constants import CFG_YML, TRAIN_PKL, TRAIN_PY, TRIAL_ID
from .storage import Storage
from .trial import Trial


class Backend:
    """The Backend is responsible for running the trials."""

    @property
    def max_concurrent_trials(self):
        """This equals the number of cpus on your machine."""
        return multiprocessing.cpu_count()

    def run(self, trial: Trial, storage: Storage) -> subprocess.Popen:
        """Run a trial in a subprocess."""
        try:
            train_path = storage.get_path(TRAIN_PY)
        except FileNotFoundError:
            # quick check if we can load the train function ...
            storage.load(TRAIN_PKL)
            train_path = storage.dump(self._train_script(storage.path), TRAIN_PY)

        cfg_path = storage.dump(trial.cfg, Path(trial.trial_id) / CFG_YML)

        args = self._build_args(train_path, cfg_path)
        out_path = cfg_path.parent / "stdout.log"
        err_path = cfg_path.parent / "stderr.log"
        os.environ[TRIAL_ID] = trial.trial_id  # logger needs it!
        with out_path.open("w") as out_file, err_path.open("w") as err_file:
            trial_process = subprocess.Popen(
                args,
                shell=True,
                executable="/bin/bash",
                cwd=cfg_path.parent,
                stderr=err_file,
                stdout=out_file,
            )

        return trial_process

    @staticmethod
    def _train_script(storage_path: Path) -> str:
        """The python script that executes the train function."""
        template = dedent(
            f"""\
            from slurm_sweeps.storage import Storage
            import sys

            storage = Storage("{storage_path}")
            train = storage.load("{TRAIN_PKL}")
            cfg = storage.load(sys.argv[1])

            train(cfg)
            """
        )

        return template

    @staticmethod
    def _build_args(train_path: Path, cfg_path: Path) -> str:
        """Build arguments for the subprocess."""
        return f"python {train_path} {cfg_path}"


class SlurmBackend(Backend):
    """Execute the training runs on a Slurm cluster via `srun`.

    Pass an instance of this class to your experiment.

    Args:
        exclusive: Add the `--exclusive` switch.
        nodes: How many nodes do you request for your srun?
        ntasks: How many tasks do you request for your srun?
        args: Additional command line arguments for srun, formatted as a string.
    """

    def __init__(
        self, exclusive: bool = True, nodes: int = 1, ntasks: int = 1, args: str = ""
    ):
        self._exclusive = exclusive
        self._nodes = nodes
        self._ntasks = ntasks
        self._args = args

    @property
    def max_concurrent_trials(self) -> int:
        """This equals the total number of SLURM tasks requested."""
        return int(int(os.environ["SLURM_NTASKS"]) / self._ntasks)

    def _build_args(self, train_path: Path, cfg_path: Path) -> str:
        """Build arguments for the subprocess."""
        slurm_cmd = (
            f"srun {'--exclusive' if self._exclusive else ''} "
            f"--nodes={self._nodes} --ntasks={self._ntasks} {self._args} "
        )
        return slurm_cmd + super()._build_args(train_path, cfg_path)

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
