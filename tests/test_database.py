import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slurm_sweeps.constants import (
    DB_CFG,
    DB_END_TIME,
    DB_ITERATION,
    DB_LOGGED,
    DB_METRIC,
    DB_METRICS,
    DB_OBJECT_DATA,
    DB_OBJECT_NAME,
    DB_START_TIME,
    DB_STATUS,
    DB_STORAGE,
    DB_TIMESTAMP,
    DB_TRIAL_ID,
    DB_TRIALS,
)
from slurm_sweeps.database import (
    Database,
    ExperimentExistsError,
    ExperimentNotFoundError,
)
from slurm_sweeps.trial import Trial


@contextmanager
def connection(path: Path):
    con = sqlite3.connect(path, isolation_level=None)
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def database(tmp_path) -> Database:
    db_path = tmp_path / "slurm_sweeps.db"
    return Database(experiment="test_experiment", path=db_path)


def test_init():
    database = Database(experiment="mock", path=".")
    assert database.path.is_absolute()


def test_exists(database):
    assert database.exists() is False

    database.create()
    assert database.exists() is True


def test_create(database):
    database.create()

    def query(table):
        with connection(database.path) as con:
            response_select = con.execute(
                f"select name from sqlite_master where type='table' and name='{table}';"
            ).fetchone()
            response_pragma = con.execute(f"pragma table_info({table})").fetchall()

        return response_select, response_pragma

    # storage table
    exists, columns = query(f"{database.experiment}{DB_STORAGE}")
    assert exists == (f"{database.experiment}{DB_STORAGE}",)
    assert [(col[1], col[2], col[4]) for col in columns] == [
        (DB_TIMESTAMP, "datetime", "strftime('%Y-%m-%d %H:%M:%f', 'NOW')"),
        (DB_OBJECT_NAME, "TEXT", None),
        (DB_OBJECT_DATA, "BLOB", None),
    ]

    # trials table
    exists, columns = query(f"{database.experiment}{DB_TRIALS}")
    assert exists == (f"{database.experiment}{DB_TRIALS}",)
    assert [(col[1], col[2], col[4]) for col in columns] == [
        (DB_TIMESTAMP, "datetime", "strftime('%Y-%m-%d %H:%M:%f', 'NOW')"),
        (DB_TRIAL_ID, "TEXT", None),
        (DB_CFG, "TEXT", None),
        (DB_START_TIME, "datetime", None),
        (DB_END_TIME, "datetime", None),
        (DB_STATUS, "TEXT", None),
    ]

    # metric table
    exists, columns = query(f"{database.experiment}{DB_METRICS}")
    assert exists == (f"{database.experiment}{DB_METRICS}",)
    assert [(col[1], col[2], col[4]) for col in columns] == [
        (DB_TIMESTAMP, "datetime", "strftime('%Y-%m-%d %H:%M:%f', 'NOW')"),
        (DB_TRIAL_ID, "TEXT", None),
        (DB_ITERATION, "INTEGER", None),
    ]

    with pytest.raises(ExperimentExistsError):
        database.create()

    database.write_metrics(trial_id="test", iteration=0, metrics={"loss": 0})
    database.write_trial(Trial({}))
    database.create(overwrite=True)

    # check if empty again

    def query(table: str):
        with connection(database.path) as con:
            response = con.execute(f"select exists (select 1 from {table});").fetchone()

        return response

    response = query(f"{database.experiment}{DB_TRIALS}")
    assert response == (0,)

    response = query(f"{database.experiment}{DB_METRICS}")
    assert response == (0,)


def test_dump_load(database):
    database.create()
    database.dump({"a": {}, "b": None})
    # check if it is replaced, not added!
    database.dump({"a": {}})

    with connection(database.path) as con:
        response = con.execute(
            f"select count(*) from {database.experiment}{DB_STORAGE}"
        ).fetchone()

    assert response[0] == 2
    assert database.load("a") == {}
    assert database.load("b") is None
    assert database.load("c") is None


def test_write_read_trials(database):
    database.create()
    cfg = {"test": {"check": 5}, "this": [1, 2, 3]}

    trial = Trial(cfg=cfg, status="completed", start_time=datetime.now())
    database.write_trial(trial)
    trials = database.read_trials()

    assert trials[0] == trial

    trial = Trial(cfg=cfg, status="pruned", end_time=datetime.now())
    database.write_trial(trial)
    trials = database.read_trials()

    assert len(trials) == 1
    assert trials[0] == trial

    process = subprocess.Popen("echo")
    trial2 = Trial(
        cfg={**cfg, "and": "that"}, start_time=datetime.now(), process=process
    )
    database.write_trial(trial2)
    trials = database.read_trials()

    assert len(trials) == 2
    assert trials[0] == trial

    assert trials[1].process is None
    trials[1].process = process
    assert trials[1] == trial2

    trials = database.read_trials(trial_id=trials[0].trial_id)
    assert len(trials) == 1
    assert trials[0] == trial


def test_write_read_metrics(database):
    with pytest.raises(ExperimentNotFoundError):
        database.write_metrics(trial_id="test", iteration=0, metrics={"loss": 0.1})

    database.create()
    database.write_metrics(
        trial_id="test", iteration=0, metrics={"test_int": 1, "test_float": 0.5}
    )

    df = database.read_metrics()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == [
        DB_TIMESTAMP,
        DB_TRIAL_ID,
        DB_ITERATION,
        f"{DB_METRIC}test_int",
        f"{DB_METRIC}test_int{DB_LOGGED}",
        f"{DB_METRIC}test_float",
        f"{DB_METRIC}test_float{DB_LOGGED}",
    ]
    assert type(df[DB_TIMESTAMP].iloc[0]) is str
    assert (
        df.iloc[:, 1:]
        .compare(
            pd.DataFrame(
                {
                    DB_TRIAL_ID: ["test"],
                    DB_ITERATION: [0],
                    f"{DB_METRIC}test_int": [1],
                    f"{DB_METRIC}test_int{DB_LOGGED}": [1],
                    f"{DB_METRIC}test_float": [0.5],
                    f"{DB_METRIC}test_float{DB_LOGGED}": [1],
                }
            )
        )
        .empty
    )
    with connection(database.path) as con:
        response = con.execute(
            f"PRAGMA table_info({database.experiment}{DB_METRICS});"
        ).fetchall()
    assert response[-4][2] == "INTEGER"
    assert response[-3][2] == "INTEGER"
    assert response[-2][2] == "REAL"
    assert response[-1][2] == "INTEGER"

    database.write_metrics(trial_id="test", iteration=1, metrics={"test_int": 1.2})
    df = database.read_metrics()
    assert len(df) == 2
    assert (
        df.iloc[:, 1:]
        .compare(
            pd.DataFrame(
                {
                    DB_TRIAL_ID: ["test", "test"],
                    DB_ITERATION: [0, 1],
                    f"{DB_METRIC}test_int": [1.0, 1.2],
                    f"{DB_METRIC}test_int{DB_LOGGED}": [1, 1],
                    f"{DB_METRIC}test_float": [0.5, np.nan],
                    f"{DB_METRIC}test_float{DB_LOGGED}": [1, np.nan],
                }
            )
        )
        .empty
    )


def read_or_write(mode: str, database: Database):
    if mode == "w":
        database.write_metrics(trial_id="test", iteration=0, metrics={"loss": 0.9})
    else:
        database.read_metrics(metric="loss")


def test_concurrent_write_read(database):
    n = 100

    database.create()
    database.write_metrics(trial_id="test", iteration=0, metrics={"loss": 0.9})

    args = np.random.choice(["w", "r"], size=n)
    with Pool(10) as p:
        p.map(
            partial(
                read_or_write,
                database=database,
            ),
            args,
        )
    df = database.read_metrics()

    assert len(df) == 1 + sum(args == "w")
    assert all([col in df.columns for col in [DB_TIMESTAMP, f"{DB_METRIC}loss"]])


def test_nan_values(database):
    database.create()

    database.write_metrics(trial_id="test", iteration=0, metrics={"loss": float("nan")})
    df = database.read_metrics()
    assert np.isnan(df[f"{DB_METRIC}loss"].iloc[0])


@pytest.mark.skip("Only for speed comparisons (OUTDATED!)")
def test_speed(monkeypatch, database):
    # TODO: Update!
    from slurm_sweeps.constants import DB_PATH, EXPERIMENT_NAME
    from slurm_sweeps.logger import Logger

    experiment = "test_speed"
    n = 500

    monkeypatch.setenv(DB_PATH, database.path)
    monkeypatch.setenv(EXPERIMENT_NAME, experiment)

    database.create(experiment)
    logger = Logger()
    for i in range(n):
        if i % 1 == 0:
            logger.log({"loss": 1}, iteration=i)
        if i % 1 == 0:
            database.read(experiment)
        if i % 50 == 0:
            logger.log({"loss2": 2}, iteration=i % 50)
            database.read(experiment)
