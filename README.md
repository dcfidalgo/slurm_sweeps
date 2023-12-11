<h1 align="center">
  <a href=""><img src="slurm_sweeps.png" alt="slurm sweeps logo" width="210"></a>
  <br>
  slurm sweeps
</h1>
<p align="center"><b>A simple tool to perform parameter sweeps on SLURM clusters.</b></p>
<p align="center">
  <a href="https://github.com/dcfidalgo/slurm_sweeps/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/dcfidalgo/slurm_sweeps.svg?color=blue">
  </a>
  <a href="https://app.codecov.io/gh/dcfidalgo/slurm_sweeps">
    <img alt="Codecov" src="https://img.shields.io/codecov/c/gh/dcfidalgo/slurm_sweeps">
  </a>
</p>

The main motivation was to provide a lightweight [ASHA implementation](https://arxiv.org/abs/1810.05934) for
[SLURM clusters](https://slurm.schedmd.com/overview.html) that is fully compatible with
[pytorch-lightning's ddp](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel).

It is heavily inspired by tools like [Ray Tune](https://www.ray.io/ray-tune) and [Optuna](https://optuna.org/).
However, on a SLURM cluster, these tools can be complicated to set up and introduce considerable overhead.

*Slurm sweeps* is simple, lightweight, and has few dependencies.
It uses SLURM Job Steps to run the individual trials.

## Installation

```commandline
pip install slurm-sweeps
```

### Dependencies
- cloudpickle
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
#SBATCH --ntasks-per-node=18
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
class Experiment(
    train: Callable,
    cfg: Dict,
    name: str = "MySweep",
    local_dir: Union[str, Path] = "./slurm-sweeps",
    asha: Optional[ASHA] = None,
    slurm_cfg: Optional[SlurmCfg] = None,
    restore: bool = False,
    overwrite: bool = False,
)
```

Set up an HPO experiment.

**Arguments**:

- `train` - A train function that takes as input the `cfg` dict.
- `cfg` - A dict passed on to the `train` function.
  It must contain the search spaces via `slurm_sweeps.Uniform`, `slurm_sweeps.Choice`, etc.
- `name` - The name of the experiment.
- `local_dir` - Where to store and run the experiments. In this directory,
  we will create the database `slurm_sweeps.db` and a folder with the experiment name.
- `slurm_cfg` - The configuration of the Slurm backend responsible for running the trials.
  We automatically choose this backend when slurm sweeps is used within an sbatch script.
- `asha` - An optional ASHA instance to cancel less promising trials.
- `restore` - Restore an experiment with the same name?
- `overwrite` - Overwrite an existing experiment with the same name?

#### `Experiment.name`

```python
@property
def name() -> str
```

The name of the experiment.

#### `Experiment.local_dir`

```python
@property
def local_dir() -> Path
```

The local directory of the experiment.

#### `Experiment.run`

```python
def run(
    n_trials: int = 1,
    max_concurrent_trials: Optional[int] = None,
    summary_interval_in_sec: float = 5.0,
    nr_of_rows_in_summary: int = 10,
    summarize_cfg_and_metrics: Union[bool, List[str]] = True
) -> pd.DataFrame
```

Run the experiment.

**Arguments**:

- `n_trials` - Number of trials to run. For grid searches, this parameter is ignored.
- `max_concurrent_trials` - The maximum number of trials running concurrently. By default, we will set this to
  the number of cpus available, or the number of total Slurm tasks divided by the number of tasks
  requested per trial.
- `summary_interval_in_sec` - Print a summary of the experiment every x seconds.
- `nr_of_rows_in_summary` - How many rows of the summary table should we print?
- `summarize_cfg_and_metrics` - Should we include the cfg and the metrics in the summary table?
  You can also pass in a list of strings to only select a few cfg and metric keys.

**Returns**:

  A summary of the trials in a pandas DataFrame.

### `CLASS slurm_sweeps.ASHA`

```python
class ASHA(
    metric: str,
    mode: str,
    reduction_factor: int = 4,
    min_t: int = 1,
    max_t: int = 50,
)
```

Basic implementation of the Asynchronous Successive Halving Algorithm (ASHA) to prune unpromising trials.

**Arguments**:

- `metric` - The metric you want to optimize.
- `mode` - Should the metric be minimized or maximized? Allowed values: ["min", "max"]
- `reduction_factor` - The reduction factor of the algorithm
- `min_t` - Minimum number of iterations before we consider pruning.
- `max_t` - Maximum number of iterations.

#### `ASHA.metric`

```python
@property
def metric() -> str
```

The metric to optimize.

#### `ASHA.mode`

```python
@property
def mode() -> str
```

The 'mode' of the metric, either 'max' or 'min'.

#### `ASHA.find_trials_to_prune`

```python
def find_trials_to_prune(database: "pd.DataFrame") -> List[str]
```

Check the database and find trials to prune.

**Arguments**:

- `database` - The experiment's metrics table of the database as a pandas DataFrame.


**Returns**:

  List of trial ids that should be pruned.

### CLASS `slurm_sweeps.SlurmCfg`

```python
@dataclass
class SlurmCfg:
  exclusive: bool = True
  nodes: int = 1
  ntasks: int = 1
  args: str = ""
```

A configuration class for the SlurmBackend.

**Arguments**:

- `exclusive` - Add the `--exclusive` switch.
- `nodes` - How many nodes do you request for your srun?
- `ntasks` - How many tasks do you request for your srun?
- `args` - Additional command line arguments for srun, formatted as a string.

### FUNCTION `slurm_sweeps.log`

```python
def log(metrics: Dict[str, Union[float, int]], iteration: int)
```

Log metrics to the database.

If ASHA is configured, this also checks if the trial needs to be pruned.

**Arguments**:

- `metrics` - A dictionary containing the metrics.
- `iteration` - Iteration of the metrics. Most of the time this will be the epoch.

**Raises**:

-  `TrialPruned` if the holy ASHA says so!
-  `TypeError` if a metric is not of type `float` or `int`.

## Contact
David Carreto Fidalgo (david.carreto.fidalgo@mpcdf.mpg.de)
