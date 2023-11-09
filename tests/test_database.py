from functools import partial
from multiprocessing import Pool
from typing import Dict, Union

import numpy as np
import pytest

from slurm_sweeps.constants import TIMESTAMP
from slurm_sweeps.database import ExperimentExistsError, FileDatabase, SqlDatabase


@pytest.fixture(params=["file", "sql"])
def database(request, tmp_path) -> Union[FileDatabase, SqlDatabase]:
    db_path = tmp_path / "slurm_sweeps.db"
    if request.param == "file":
        return FileDatabase(db_path)
    if request.param == "sql":
        return SqlDatabase(db_path)
    raise NotImplementedError(
        f"'database' fixture not implemented for '{request.param}'"
    )


@pytest.mark.parametrize(
    "database, is_file_or_dir",
    [("file", "is_dir"), ("sql", "is_file")],
    indirect=["database"],
)
def test_init(database, is_file_or_dir):
    assert getattr(database.path, is_file_or_dir)()


def test_fasteners_not_installed(monkeypatch):
    monkeypatch.setattr("slurm_sweeps.database._has_fasteners", False)
    with pytest.raises(ModuleNotFoundError):
        FileDatabase()


def test_create(database):
    experiment = "test_experiment"
    database.create(experiment)
    database.write(experiment, {"test": "test"})
    assert len(database.read(experiment)) == 1

    with pytest.raises(ExperimentExistsError):
        database.create(experiment)

    database.create(experiment, overwrite=True)
    assert len(database.read(experiment)) == 0


def read_or_write(
    mode: str, database: Union[FileDatabase, SqlDatabase], experiment: str, row: Dict
):
    if mode == "w":
        database.write(experiment, row)
    else:
        database.read(experiment)


@pytest.mark.parametrize("database", ["file", "sql"], indirect=["database"])
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
    assert all([col in df.columns for col in [TIMESTAMP, "loss", "lr"]])


def test_nan_values(database):
    experiment = "test_db_nan"
    database.create(experiment)

    database.write(experiment, {"loss": float("nan")})
    df = database.read(experiment)
    print(df)
    assert np.isnan(df["loss"].iloc[0])


@pytest.mark.skip("Only for speed comparisons")
def test_speed(monkeypatch, database):
    from slurm_sweeps import Logger
    from slurm_sweeps.constants import DB_PATH, EXPERIMENT_NAME

    experiment = "test_speed"
    n = 500

    monkeypatch.setenv(DB_PATH, database.path)
    monkeypatch.setenv(EXPERIMENT_NAME, experiment)

    database.create(experiment)
    logger = Logger({})
    for i in range(n):
        if i % 1 == 0:
            logger.log({"loss": 1}, iteration=i)
        if i % 1 == 0:
            database.read(experiment)
        if i % 50 == 0:
            logger.log({"loss2": 2}, iteration=i % 50)
            database.read(experiment)
