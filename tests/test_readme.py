import subprocess
from time import sleep

import pytest

from slurm_sweeps.constants import DB_ITERATION
from slurm_sweeps.database import SqlDatabase


def is_slurm_available() -> bool:
    try:
        subprocess.check_output(["sinfo"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    else:
        return True


def test_readme_example_on_local(tmp_path):
    from time import sleep

    import slurm_sweeps as ss

    # Define your train function
    def train(cfg: dict):
        for epoch in range(cfg["epochs"]):
            sleep(0.5)
            loss = (cfg["parameter"] - 1) ** 2 * epoch
            # log your metrics
            ss.log({"loss": loss}, epoch)

    # Define your experiment
    experiment = ss.Experiment(
        train=train,
        cfg={
            "epochs": 10,
            "parameter": ss.Uniform(0, 2),
        },
        local_dir=tmp_path / "slurm_sweeps",
        asha=ss.ASHA(metric="loss", mode="min"),
    )

    # Run your experiment
    dataframe = experiment.run(n_trials=10)

    # Your results are stored in a pandas DataFrame
    print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")

    assert len(dataframe) > 10
    assert dataframe[DB_ITERATION].sort_values().iloc[-1] == 9


@pytest.mark.skipif(not is_slurm_available(), reason="requires a SLURM cluster")
def test_readme_example_on_slurm(tmp_path):
    """This test is meant for our CI on GitHub where we set up a SLURM cluster"""
    local_dir = tmp_path / "slurm_sweeps"

    python_script = f"""import slurm_sweeps as ss
from time import sleep

def train(cfg: dict):
    for epoch in range(cfg["epochs"]):
        sleep(0.5)
        loss = (cfg["parameter"] - 1) ** 2 * epoch
        # log your metrics
        ss.log({{"loss": loss}}, epoch)

# Define your experiment
experiment = ss.Experiment(
    train=train,
    cfg={{
"epochs": 10,
        "parameter": ss.Uniform(0, 2),
    }},
    local_dir="{local_dir}",
    asha=ss.ASHA(metric="loss", mode="min"),
)

# Run your experiment
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

    # Relax until issue with slurm GitHub action is fixed: https://github.com/koesterlab/setup-slurm-action/issues/4
    assert ("max number of concurrent trials: 2" in job_out.decode()[50:]) or (
        "max number of concurrent trials: 4" in job_out.decode()[50:]
    )
    assert len(dataframe) > 10
    assert dataframe[DB_ITERATION].sort_values().iloc[-1] == 9
