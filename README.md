# slurm sweeps
A simple tool to perform sweeps on SLURM clusters.

The main idea is to provide a lightweight [ASHA implementation](https://arxiv.org/abs/1810.05934) for
[SLURM clusters](https://slurm.schedmd.com/overview.html) that is fully compatible with
[pytorch-lightning's ddp](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel).

Heavily inspired by tools like [Ray Tune](https://www.ray.io/ray-tune) and [Optuna](https://optuna.org/).
However, on a SLURM cluster these tools can be complicated to set up and introduce considerable overhead.
That was the main motivation for "slurm sweeps".

## Installation
Clone the repo, change the directory and install it via pip:

```commandline
cd slurm_sweeps
pip install .
```

### Dependencies
- cloudpickle
- fasteners
- numpy
- pandas
- pyyaml

## Usage
Run this example on your laptop.
By default, each CPU runs a trial in parallel.

```python
""" Content of test_ss.py """
from time import sleep
import slurm_sweeps as ss


def train(cfg):
    logger = ss.Logger(cfg)
    for epoch in range(1, 10):
        sleep(0.5)
        loss = (cfg["parameter"] - 1) ** 2 * epoch
        logger.log("loss", loss, epoch)


experiment = ss.Experiment(
    train=train,
    cfg={
        "parameter": ss.Uniform(0, 2),
    },
    asha=ss.ASHA(metric="loss", mode="min"),
    exist_ok=True
)


dataframe = experiment.run(n_trials=1000)

print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")
```

Or submit it to a SLURM cluster.
By default, this will run `$SLURM_NTASKS` trials in parallel.
In the case below: 2 nodes * 18 tasks = 36 trials.

Batch script `test_ss.slurm`:
```bash
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --tasks-per-node=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1GB

python test_ss.py
```
```commandline
sbatch test_ss.slurm
```

See the `tests` folder for an advanced example of training a PyTorch model with Lightning's DDP.

## Documentation
...

## Contact
David Carreto Fidalgo (david.carreto.fidalgo@mpcdf.mpg.de)
