import os
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment

from slurm_sweeps import Logger


class SlurmSweepsCallback(pl.Callback):
    """A Pytorch Lightning callback that logs the metric to slurm sweeps.

    Args:
        cfg: The cfg dict of the train function.
        metric: The metric you want to optimize.
        log_to_wandb: Log to WandB? By default, we automatically log to Wandb if it is installed.
        wandb_init_kwargs: Passed on to the wandb.init() method
    """

    def __init__(
        self,
        cfg: Dict,
        metric: str,
        log_to_wandb: Optional[bool] = None,
        wandb_init_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self._logger = Logger(cfg)
        self._metric = metric
        self._wandb_init_kwargs = wandb_init_kwargs or {}

        if log_to_wandb is False:
            self._wandb = None
        else:
            try:
                import wandb
            except ImportError:
                self._wandb = None
            else:
                self._wandb = wandb

    @property
    def logger(self):
        return self._logger

    @pl.utilities.rank_zero.rank_zero_only
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        if self._wandb:
            self._wandb.init(
                name=self._logger.trial.trial_id,
                config=self._logger.trial.cfg,
                **self._wandb_init_kwargs
            )

    @pl.utilities.rank_zero.rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not trainer.sanity_checking:
            iteration = trainer.current_epoch

            if self._wandb is not None:
                self._wandb.log(trainer.callback_metrics, step=iteration)

            metric = float(trainer.callback_metrics[self._metric].detach().cpu())
            # Keep in mind, the log call can raise an Exception if ASHA says so,
            # so it should come after the wandb.log call!
            self.logger.log(self._metric, metric, iteration)


class ParallelSlurmEnvironment(SLURMEnvironment):
    @property
    def main_address(self) -> str:
        if "MASTER_ADDR" in os.environ:
            return os.environ["MASTER_ADDR"]
        return super().main_address
