from slurm_sweeps.trial import Trial


def test_init():
    trial = Trial(cfg={})
    assert trial.status is None
