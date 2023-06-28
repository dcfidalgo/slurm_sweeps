import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pytest

from slurm_sweeps import Database


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(tmp_path / ".db")


@pytest.mark.parametrize("path_or_str", ["path", "str"])
def test_init(tmp_path, path_or_str):
    db_path = tmp_path / "test_init"
    if path_or_str == "str":
        db_path = str(db_path)
    Database(db_path)

    assert Path(db_path).is_dir()


@pytest.mark.parametrize("exist_ok", [True, False], ids=["exist_ok", "exist_not_ok"])
def test_create(database, exist_ok):
    if exist_ok:
        database.create("test_db_exp", exist_ok=exist_ok)
        txt = database.path / "test_db_exp.txt"
        assert txt.exists()
    else:
        (database.path / "test_db_exp.txt").touch()
        with pytest.raises(FileExistsError):
            database.create("test_db_exp", exist_ok=exist_ok)


def test_write_read(database):
    n = 1000
    args = ["test_db_write_read" for _ in range(n)]
    with Pool(10) as p:
        p.map(partial(database.write, row={"loss": 1, "a": 2}), args)
    with Pool(10) as p:
        dfs = p.map(database.read, args)

    assert all([len(df) == n for df in dfs])
    assert dict(dfs[0].iloc[0]) == {"a": 2, "loss": 1}


def test_nan(database):
    database.write("test_db_nan", {"loss": float("nan")})
    df = database.read("test_db_nan")
    assert math.isnan(df["loss"][0])
