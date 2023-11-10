import subprocess
from time import sleep

import pytest

import slurm_sweeps as ss
from slurm_sweeps import SqlDatabase
from slurm_sweeps.constants import ITERATION


def is_slurm_available() -> bool:
    try:
        subprocess.check_output(["sinfo"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    else:
        return True


def test_readme_example_on_local(tmp_path):
    def train(cfg):
        logger = ss.Logger(cfg)
        for epoch in range(1, 10):
            sleep(0.5)
            loss = (cfg["parameter"] - 1) ** 2 * epoch
            logger.log({"loss": loss}, epoch)

    experiment = ss.Experiment(
        train=train,
        local_dir=tmp_path / "slurm_sweeps",
        cfg={
            "parameter": ss.Uniform(0, 2),
        },
        asha=ss.ASHA(metric="loss", mode="min"),
    )

    dataframe = experiment.run(n_trials=10)

    print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")

    assert len(dataframe) > 10
    assert dataframe[ITERATION].sort_values().iloc[-1] == 9


@pytest.mark.skipif(not is_slurm_available(), reason="requires a SLURM cluster")
def test_readme_example_on_slurm(tmp_path):
    """This test is meant for our CI on GitHub where we set up a SLURM cluster"""
    local_dir = tmp_path / "slurm_sweeps"

    python_script = f"""import slurm_sweeps as ss
from time import sleep

def train(cfg):
    logger = ss.Logger(cfg)
    for epoch in range(1, 10):
        sleep(0.5)
        loss = (cfg["parameter"] - 1) ** 2 * epoch
        logger.log({{"loss": loss}}, epoch)

experiment = ss.Experiment(
    train=train,
    local_dir="{local_dir}",
    cfg={{
        "parameter": ss.Uniform(0, 2),
    }},
    asha=ss.ASHA(metric="loss", mode="min"),
)

dataframe = experiment.run(n_trials=10)
"""
    with (tmp_path / "train.py").open("w") as file:
        file.write(python_script)

    slurm_script = f"""#!/bin/bash -l
#SBATCH -J train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

python train.py
"""
    with (tmp_path / "train.slurm").open("w") as file:
        file.write(slurm_script)

    subprocess.check_output(["sbatch", "train.slurm"], cwd=tmp_path)

    # wait for job to finish
    running = True
    while running:
        sleep(1)
        squeue = subprocess.check_output(["squeue", "--me"])
        if "train" not in squeue.decode():
            running = False

    # check output
    job_out = subprocess.check_output(["cat", "slurm-3.out"], cwd=tmp_path)
    dataframe = SqlDatabase(local_dir / "slurm_sweeps.db").read("MySweep")

    assert "max number of concurrent trials: 2" in job_out.decode()[50:]
    assert len(dataframe) > 10
    assert dataframe[ITERATION].sort_values().iloc[-1] == 9
