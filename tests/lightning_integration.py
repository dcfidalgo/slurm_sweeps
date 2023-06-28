import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import slurm_sweeps as ss
from slurm_sweeps.integrations.lightning import (
    ParallelSlurmEnvironment,
    SlurmSweepsCallback,
)


class Model(pl.LightningModule):
    def __init__(self, lr: float, **kwargs):
        super().__init__()
        self._lr = lr
        self._net = nn.Sequential(nn.Linear(28 * 28, 32), nn.ReLU(), nn.Linear(32, 10))

        self.save_hyperparameters()

    def forward(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        x = self._net(x)
        loss = F.cross_entropy(x, y)

        return loss

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer


def train(cfg):
    dataset = MNIST(cfg["data_path"], download=True, transform=transforms.ToTensor())
    dataset_val = MNIST(
        cfg["data_path"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(dataset, batch_size=cfg["batch_size"])
    val_loader = DataLoader(dataset_val, batch_size=128)

    # model
    model = Model(lr=cfg["lr"])

    plugins = None
    if cfg.get("slurm"):
        plugins = ParallelSlurmEnvironment(auto_requeue=False)
        os.environ["MASTER_ADDR"] = "localhost"

    # train model
    trainer = pl.Trainer(
        strategy="ddp",
        devices=4,
        accelerator="cpu",
        max_epochs=cfg["max_epochs"],
        callbacks=SlurmSweepsCallback(
            cfg,
            "val_loss",
            log_to_wandb=False,
            wandb_init_kwargs=dict(
                project=cfg["wandb_project"], group=cfg["wandb_group"]
            ),
        ),
        enable_progress_bar=True,
        plugins=plugins,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    cfg = {
        "data_path": "/home/david/mpcdf/slurm_sweeps/tests",
        "lr": ss.LogUniform(1.0e-3, 1.0e-1),
        "batch_size": 128,
        "max_epochs": 1,
        "wandb_project": "slurm_sweeps",
        "wandb_group": "test",
        "slurm": True,
    }

    experiment = ss.Experiment(
        train=train,
        cfg=cfg,
        name="LightningIntegration_task1",
        asha=ss.ASHA(metric="val_loss", mode="min", reduction_factor=2, max_t=2),
        exist_ok=True,
    )

    experiment.run(
        4,
        max_concurrent_trials=2,
        summarize_cfg_and_metrics=["lr", "batch_size", "val_loss"],
    )
