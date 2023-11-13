import pandas as pd
import pytest

from slurm_sweeps import ASHA
from slurm_sweeps.constants import DB_ITERATION, DB_TRIAL_ID


@pytest.fixture(
    params=[0, 1]
)  # make sure it works with ITERATION starting with 0 and 1
def database(request):
    data = [
        {DB_ITERATION: 0 + request.param, DB_TRIAL_ID: "a", "loss": 10.0},
        {DB_ITERATION: 0 + request.param, DB_TRIAL_ID: "b", "loss": 9.0},
        {DB_ITERATION: 0 + request.param, DB_TRIAL_ID: "c", "loss": 8.0},
        {DB_ITERATION: 0 + request.param, DB_TRIAL_ID: "d", "loss": 7.0},
        {DB_ITERATION: 1 + request.param, DB_TRIAL_ID: "aa", "loss": 10.0},
        {DB_ITERATION: 1 + request.param, DB_TRIAL_ID: "bb", "loss": 9.0},
        {DB_ITERATION: 1 + request.param, DB_TRIAL_ID: "cc", "loss": 8.0},
        {DB_ITERATION: 1 + request.param, DB_TRIAL_ID: "dd", "loss": 7.0},
        {DB_ITERATION: 1 + request.param, DB_TRIAL_ID: "nan", "loss": float("nan")},
        {DB_ITERATION: 3 + request.param, DB_TRIAL_ID: "aaa", "loss": 10.0},
        {DB_ITERATION: 3 + request.param, DB_TRIAL_ID: "bbb", "loss": 9.0},
        {DB_ITERATION: 3 + request.param, DB_TRIAL_ID: "ccc", "loss": 8.0},
        {DB_ITERATION: 3 + request.param, DB_TRIAL_ID: "ddd", "loss": 7.0},
    ]
    return pd.DataFrame(data)


def test_asha_init():
    with pytest.raises(AssertionError):
        ASHA(metric="loss", mode="not min or max")
    with pytest.raises(AssertionError):
        ASHA(metric="loss", mode="min", reduction_factor=1)
    with pytest.raises(AssertionError):
        ASHA(metric="loss", mode="min", min_t=0)
    with pytest.raises(AssertionError):
        ASHA(metric="loss", mode="min", min_t=2, max_t=1)

    asha = ASHA(metric="loss", mode="min")
    assert asha._rungs == [16, 4, 1]

    asha = ASHA(metric="loss", mode="min", min_t=2, max_t=16, reduction_factor=2)
    assert asha._rungs == [16, 8, 4, 2]


@pytest.mark.parametrize(
    "rf, expected",
    [
        (2, ["aaa", "bbb", "aa", "bb", "nan", "a", "b"]),
        (4, ["aaa", "bbb", "ccc", "a", "b", "c"]),
    ],
)
def test_asha_rf(database, rf, expected):
    asha = ASHA(metric="loss", mode="min", reduction_factor=rf)
    assert asha.find_trials_to_prune(database) == expected
