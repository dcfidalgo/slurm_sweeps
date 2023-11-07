import subprocess
from time import sleep

import pytest

import slurm_sweeps as ss
from slurm_sweeps import Database, SlurmBackend


def test_readme_example_on_local(tmp_path):
    def train(cfg):
        logger = ss.Logger(cfg)
        for epoch in range(1, 10):
            sleep(0.5)
            loss = (cfg["parameter"] - 1) ** 2 * epoch
            logger.log("loss", loss, epoch)

    experiment = ss.Experiment(
        train=train,
        local_dir=tmp_path / "slurm_sweeps",
        cfg={
            "parameter": ss.Uniform(0, 2),
        },
        asha=ss.ASHA(metric="loss", mode="min"),
    )

    dataframe = experiment.run(n_trials=20)

    print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")

    assert len(dataframe) > 20
    assert dataframe["iteration"].sort_values().iloc[-1] == 9


@pytest.mark.skipif(not SlurmBackend.is_available(), reason="requires a SLURM cluster")
def test_readme_example_on_slurm(tmp_path):
    local_dir = tmp_path / "slurm_sweeps"
    python_script = f"""import slurm_sweeps as ss
from time import sleep

def train(cfg):
    logger = ss.Logger(cfg)
    for epoch in range(1, 10):
        sleep(0.5)
        loss = (cfg["parameter"] - 1) ** 2 * epoch
        logger.log("loss", loss, epoch)

experiment = ss.Experiment(
    train=train,
    local_dir={local_dir},
    cfg={{
        "parameter": ss.Uniform(0, 2),
    }},
    asha=ss.ASHA(metric="loss", mode="min"),
)

dataframe = experiment.run(n_trials=20)
"""
    with (tmp_path / "train.py").open("w") as file:
        file.write(python_script)

    slurm_script = f"""#!/bin/bash -l
#SBATCH -J train
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1

python train.py
"""
    with (tmp_path / "train.slurm").open("w") as file:
        file.write(slurm_script)

    subprocess.check_output(["sbatch", "train.slurm"], cwd=tmp_path)

    running = True
    while running:
        squeue = subprocess.check_output(["squeue", "--me"])
        if "train" not in squeue.decode():
            running = False

    dataframe = Database(local_dir / "slurm_sweeps.db").read("MySweep")

    assert len(dataframe) > 20
    assert dataframe["iteration"].sort_values().iloc[-1] == 9
