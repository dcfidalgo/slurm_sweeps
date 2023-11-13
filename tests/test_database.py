import sqlite3
from functools import partial
from multiprocessing import Pool
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from slurm_sweeps.constants import DB_CFG, DB_ITERATION, DB_TIMESTAMP, DB_TRIAL_ID
from slurm_sweeps.database import ExperimentExistsError, SqlDatabase


@pytest.fixture
def database(tmp_path) -> SqlDatabase:
    db_path = tmp_path / "slurm_sweeps.db"
    return SqlDatabase(db_path)


def test_init(database):
    assert database.path.is_file()


def test_create(database):
    experiment = "test_experiment"
    database.create(experiment)

    con = sqlite3.connect(database.path)
    check_exists = con.execute(
        "select name from sqlite_master where type='table' and name='test_experiment';"
    ).fetchone()
    check_columns = con.execute(f"pragma table_info({experiment})").fetchall()
    con.close()
    assert check_exists == ("test_experiment",)
    assert [(col[1], col[2], col[4]) for col in check_columns] == [
        (DB_TIMESTAMP, "datetime", "strftime('%Y-%m-%d %H:%M:%f', 'NOW')"),
        (DB_TRIAL_ID, "TEXT", None),
        (DB_ITERATION, "INTEGER", None),
        (DB_CFG, "TEXT", None),
    ]

    with pytest.raises(ExperimentExistsError):
        database.create(experiment)

    database.write("test_experiment", {"test": 0})
    database.create(experiment, overwrite=True)
    # check if empty again
    con = sqlite3.connect(database.path)
    response = con.execute("select exists (select 1 from test_experiment);").fetchone()
    con.close()
    assert response == (0,)


def test_write_and_read(database):
    database.create("test_experiment")
    database.write("test_experiment", {"test_int": 1, "test_float": 0.5})

    con = sqlite3.connect(database.path)
    response = con.execute("select count(*) from test_experiment").fetchone()
    con.close()
    assert response == (1,)

    df = database.read("test_experiment")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        DB_TIMESTAMP,
        DB_TRIAL_ID,
        DB_ITERATION,
        DB_CFG,
        "test_int",
        "test_float",
    ]
    assert type(df[DB_TIMESTAMP].iloc[0]) is str
    assert (
        df.iloc[:, 1:]
        .compare(
            pd.DataFrame(
                {
                    DB_TRIAL_ID: [np.nan],
                    DB_ITERATION: [np.nan],
                    DB_CFG: [np.nan],
                    "test_int": [1],
                    "test_float": [0.5],
                }
            )
        )
        .empty
    )


def read_or_write(mode: str, database: SqlDatabase, experiment: str, row: Dict):
    if mode == "w":
        database.write(experiment, row)
    else:
        database.read(experiment)


def test_concurrent_write_read(database):
    experiment = "test_db_write_read"
    n = 250

    database.create(experiment)
    args = np.random.choice(["w", "r"], size=n)
    with Pool(10) as p:
        p.map(
            partial(
                read_or_write,
                database=database,
                experiment=experiment,
                row={"loss": 0.2, "lr": 2},
            ),
            args,
        )
    df = database.read(experiment)

    assert len(df) == sum(args == "w")
    assert all([col in df.columns for col in [DB_TIMESTAMP, "loss", "lr"]])


def test_nan_values(database):
    experiment = "test_db_nan"
    database.create(experiment)

    database.write(experiment, {"loss": float("nan")})
    df = database.read(experiment)
    print(df)
    assert np.isnan(df["loss"].iloc[0])


@pytest.mark.skip("Only for speed comparisons")
def test_speed(monkeypatch, database):
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
