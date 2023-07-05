# slurm sweeps
A simple tool to perform parameter sweeps on SLURM clusters.

The main idea is to provide a lightweight [ASHA implementation](https://arxiv.org/abs/1810.05934) for
[SLURM clusters](https://slurm.schedmd.com/overview.html) that is fully compatible with
[pytorch-lightning's ddp](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel).

It is heavily inspired by tools like [Ray Tune](https://www.ray.io/ray-tune) and [Optuna](https://optuna.org/).
However, on a SLURM cluster, these tools can be complicated to set up and introduce considerable overhead.

*Slurm sweeps* is simple, lightweight, and has few dependencies.
It uses SLURM Job Steps to run the individual trials.

## Installation
Clone the repo and install it via pip:

```commandline
pip install .
```

### Dependencies
- cloudpickle
- fasteners
- numpy
- pandas
- pyyaml

## Usage
You can just run this example on your laptop.
By default, the maximum number of parallel trials equals the number of CPUs on your machine.

```python
""" Content of test_ss.py """
from time import sleep
import slurm_sweeps as ss


# Define your train function
def train(cfg: dict):
    logger = ss.Logger(cfg)
    for epoch in range(cfg["epochs"]):
        sleep(0.5)
        loss = (cfg["parameter"] - 1) ** 2 * epoch
        logger.log("loss", loss, epoch)


# Define your experiment
experiment = ss.Experiment(
    train=train,
    cfg={
        "epochs": 10,
        "parameter": ss.Uniform(0, 2),
    },
    asha=ss.ASHA(metric="loss", mode="min"),
)


# Run your experiment
dataframe = experiment.run(n_trials=1000)

# Your results are stored in a pandas DataFrame
print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")
```

Or submit it to a SLURM cluster.
Write a small SLURM script `test_ss.slurm` that runs the code above:
```bash
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --tasks-per-node=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1GB

python test_ss.py
```

By default, this will run `$SLURM_NTASKS` trials in parallel.
In the case above: `2 nodes * 18 tasks = 36 trials`

Then submit it to the queue:
```commandline
sbatch test_ss.slurm
```

See the `tests` folder for an advanced example of training a PyTorch model with Lightning's DDP.

## API Documentation

### `CLASS slurm_sweeps.Experiment`

```python
def __init__(
    train: Callable,
    cfg: Dict,
    name: str = "MySweep",
    local_dir: str = "./slurm_sweeps",
    backend: Optional[Backend] = None,
    asha: Optional[ASHA] = None,
    database: Optional[Database] = None,
    restore: bool = False,
    exist_ok: bool = False,
)
```

Run an HPO experiment using random search and the Asynchronous Successive Halving Algorithm (ASHA).

**Arguments**:

- `train` - A train function that takes as input a `cfg` dict.
- `cfg` - A dict passed on to the `train` function.
  It must contain the search spaces via `slurm_sweeps.Uniform`, `slurm_sweeps.Choice`, etc.
- `name` - The name of the experiment.
- `local_dir` - Where to store and run the experiments. In this directory
  we will create a folder with the experiment name.
- `backend` - A backend to execute the trials. By default, we choose the `SlurmBackend` if Slurm is available,
  otherwise we choose the standard `Backend` that simply executes the trial in another process.
- `asha` - An optional ASHA instance to cancel less promising trials. By default, it is None.
- `database` - A database instance to store the trial's (intermediate) results.
  By default, it will create the database at `{local_dir}/.database'.
- `restore` - Restore an experiment with the same name?
- `exist_ok` - Replace an existing experiment with the same name?

<a id="slurm_sweeps.experiment.Experiment.run"></a>

#### `Experiment.run`

```python
def run(
    n_trials: int = 1,
    max_concurrent_trials: Optional[int] = None,
    summary_interval_in_sec: float = 5.0,
    nr_of_rows_in_summary: int = 10,
    summarize_cfg_and_metrics: Union[bool, List[str]] = True
) -> pandas.DataFrame
```

Run the experiment.

**Arguments**:

- `n_trials` - Number of trials to run.
- `max_concurrent_trials` - The maximum number of trials running concurrently. By default, we will set this to
  the number of cpus available, or the number of total Slurm tasks divided by the number of trial Slurm
  tasks requested.
- `summary_interval_in_sec` - Print a summary of the experiment every x seconds.
- `nr_of_rows_in_summary` - How many rows of the summary table should we print?
- `summarize_cfg_and_metrics` - Should we include the cfg and the metrics in the summary table?
  You can also pass in a list of strings to only select a few cfg and metric keys.


**Returns**:

  A DataFrame of the database.

## Contact
David Carreto Fidalgo (david.carreto.fidalgo@mpcdf.mpg.de)
