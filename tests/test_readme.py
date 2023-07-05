from time import sleep

import slurm_sweeps as ss


def test_readme():
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
        exist_ok=True,
    )

    dataframe = experiment.run(n_trials=20)

    print(f"\nBest trial:\n{dataframe.sort_values('loss').iloc[0]}")
