import logging
from time import time
from typing import Dict

import numpy as np
from rich.progress import track

import slurm_sweeps as ss


def main(dim: int = 3, nr_trials: int = 10, use_tpe=True):
    def sphere(cfg: Dict):
        values = np.array(list(cfg.values()))
        values *= values

        ss.log({"loss": float(np.sum(values))}, iteration=1)

    experiment = ss.Experiment(
        train=sphere,
        cfg={f"x{d}": ss.Uniform(-5, 5) for d in range(dim)},
        sweep_config=ss.SweepConfig("loss", "min", use_asha=False, use_tpe=use_tpe),
        overwrite=True,
    )

    result = experiment.run(n_trials=nr_trials, max_concurrent_trials=2)
    best_trial = result.best_trial(metric="loss", mode="min")

    return best_trial.metrics["loss"][1]


if __name__ == "__main__":
    logger = logging.getLogger("slurm_sweeps")
    logger.setLevel(logging.ERROR)

    r = 10
    rdm_vals, rdm_tms, tpe_vals, tpe_tms = (
        np.empty(r),
        np.empty(r),
        np.empty(r),
        np.empty(r),
    )
    for i in track(list(range(r))):
        start = time()
        tpe_vals[i] = main(use_tpe=ss.TpeConfig(n_i=5))
        tpe_tms[i] = time() - start

        start = time()
        rdm_vals[i] = main(use_tpe=False)
        rdm_tms[i] = time() - start

    print("rdm", rdm_vals.mean(), rdm_vals.std(), rdm_tms.mean(), rdm_tms.std())
    print("tpe", tpe_vals.mean(), tpe_vals.std(), tpe_tms.mean(), tpe_tms.std())
