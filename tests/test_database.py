import math
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np
import pytest

from slurm_sweeps.database import ExperimentExistsError, FileDatabase, SQLDatabase


@pytest.fixture(params=["file", "sql"])
def database(request, tmp_path) -> Union[FileDatabase, SQLDatabase]:
    db_path = tmp_path / "slurm_sweeps.db"
    if request.param == "file":
        return FileDatabase(db_path)
    if request.param == "sql":
        return SQLDatabase(db_path)
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
    mode: str, database: Union[FileDatabase, SQLDatabase], experiment: str
):
    if mode == "w":
        database.write(
            experiment,
            {"trial_id": "012abcdfghijkl", "iteration": 1, "loss": 0.9, "lr": 0.01},
        )
    else:
        database.read(experiment)


@pytest.mark.parametrize("database", ["file", "sql"], indirect=["database"])
def test_concurrent_write_read(database):
    experiment = "test_db_write_read"
    n = 500

    database.create(experiment)
    args = np.random.choice(["w", "r"], size=n)
    with Pool(10) as p:
        p.map(partial(read_or_write, experiment=experiment, database=database), args)
    df = database.read(experiment)

    assert len(df) == sum(args == "w")
    assert all(
        [
            col in df.columns
            for col in ["timestamp", "trial_id", "iteration", "loss", "lr"]
        ]
    )


def test_nan(database):
    database.create("test_db_nan")
    database.write("test_db_nan", {"loss": float("nan")})
    df = database.read("test_db_nan")
    assert math.isnan(df["loss"][0])
