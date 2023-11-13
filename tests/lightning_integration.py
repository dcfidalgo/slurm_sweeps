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

    # set up environment for ddp
    plugins = ParallelSlurmEnvironment(auto_requeue=False)
    os.environ["MASTER_ADDR"] = "localhost"

    # model
    model = Model(lr=cfg["lr"])

    # train model
    trainer = pl.Trainer(
        strategy="ddp",
        devices=4,
        accelerator="gpu",
        max_epochs=cfg["max_epochs"],
        callbacks=SlurmSweepsCallback(
            "val_loss",
            log_to_wandb=True,
            wandb_init_kwargs=dict(
                project=cfg["wandb_project"], group=cfg["wandb_group"]
            ),
        ),
        plugins=plugins,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    experiment = ss.Experiment(
        train=train,
        cfg={
            "data_path": "/home/david/slurm_sweeps/tests",
            "lr": ss.LogUniform(1.0e-3, 1.0e-1),
            "batch_size": 128,
            "max_epochs": 2,
            "wandb_project": "slurm_sweeps",
            "wandb_group": "lightning_test",
        },
        name="LightningIntegration",
        asha=ss.ASHA(metric="val_loss", mode="min", reduction_factor=2, max_t=2),
        backend=ss.SlurmBackend(nodes=1, ntasks=4),
    )

    experiment.run(
        n_trials=4,
        summarize_cfg_and_metrics=["lr", "batch_size", "val_loss"],
    )
