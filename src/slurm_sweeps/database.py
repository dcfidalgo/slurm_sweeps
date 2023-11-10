import abc
import datetime
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from .constants import CFG, ITERATION, TIMESTAMP, TRIAL_ID

try:
    import fasteners
except ModuleNotFoundError:
    _has_fasteners = False
else:
    _has_fasteners = True


class Database(abc.ABC):
    def __init__(self, path: Union[str, Path] = "./slurm_sweeps.db"):
        self._path = Path(path).resolve()

    @property
    def path(self):
        return self._path

    @abc.abstractmethod
    def create(self, experiment: str, overwrite: bool = False):
        pass

    @abc.abstractmethod
    def write(self, experiment: str, row: Dict):
        pass

    @abc.abstractmethod
    def read(self, experiment: str) -> pd.DataFrame:
        pass


class FileDatabase(Database):
    def __init__(self, path: Union[str, Path] = "./slurm_sweeps.db"):
        if not _has_fasteners:
            raise ModuleNotFoundError(
                "You need to install 'fasteners' to use the FileDatabase: "
                "`pip install fasteners`"
            )

        super().__init__(path=path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_file_path_and_lock(
        self, experiment
    ) -> Tuple[Path, fasteners.InterProcessReaderWriterLock]:
        path = self.path / f"{experiment}.txt"
        lock = fasteners.InterProcessReaderWriterLock(f"{path}.lock")

        return path, lock

    def create(self, experiment: str, overwrite: bool = False):
        path, _ = self._get_file_path_and_lock(experiment)
        try:
            path.touch(exist_ok=overwrite)
        except FileExistsError as err:
            raise ExperimentExistsError(experiment) from err
        with path.open(mode="w"):
            pass

    def write(self, experiment: str, row: Dict):
        path, lock = self._get_file_path_and_lock(experiment)
        # quick check if file exists
        with path.open():
            pass

        row[TIMESTAMP] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        json_str = json.dumps(row)

        lock.acquire_write_lock()
        with path.open(mode="a") as f:
            f.write(json_str + "\n")
        lock.release_write_lock()

    def read(self, experiment: str) -> pd.DataFrame:
        path, lock = self._get_file_path_and_lock(experiment)
        lock.acquire_read_lock()
        database_df = pd.read_json(path, lines=True)
        lock.release_read_lock()

        return database_df


class SqlDatabase(Database):
    def __init__(self, path: Union[str, Path] = "./slurm_sweeps.db"):
        super().__init__(path)
        if not self.path.exists():
            with self._connection() as con:
                con.execute("vacuum")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            yield connection
        finally:
            connection.close()

    def create(self, experiment: str, overwrite: bool = False):
        with self._connection() as con:
            if overwrite:
                con.execute(f"drop table if exists {experiment};")
            try:
                con.execute(
                    f"create table {experiment}("
                    f"{TIMESTAMP} datetime default(strftime('%Y-%m-%d %H:%M:%f', 'NOW')),"
                    f"{TRIAL_ID} text,"
                    f"{ITERATION} integer,"
                    f"{CFG} text);"
                )
            except sqlite3.OperationalError as err:
                if "already exists" in str(err):
                    raise ExperimentExistsError(experiment) from err
                raise err

    def write(self, experiment: str, row: Dict):
        with self._connection() as con:
            # add missing columns
            missing_columns_and_types = self._get_missing_columns_and_types(
                con, experiment, row
            )
            for col, typ in missing_columns_and_types:
                try:
                    con.execute(f"alter table {experiment} add column {col} {typ}")
                # Just as a safeguard, during tests I sometimes encountered this error ...
                except sqlite3.OperationalError as err:
                    if "duplicate column name" not in str(err):
                        raise err

            # insert row
            con.execute(
                f"insert into {experiment} "
                f"({', '.join(row.keys())}) values ({', '.join(['?' for _ in range(len(row))])});",
                list(row.values()),
            )

    @staticmethod
    def _get_missing_columns_and_types(
        connection: sqlite3.Connection, experiment: str, row: Dict
    ) -> List[Tuple[str, str]]:
        def get_sql_type(value: Union[float, int]) -> str:
            if type(value) is int:
                return "integer"
            if type(value) is float:
                return "real"
            return "numeric"

        response = connection.execute(f"pragma table_info({experiment})").fetchall()
        existing_columns = [col[1].lower() for col in response]
        missing_columns = [
            (col.lower(), get_sql_type(val))
            for col, val in row.items()
            if col.lower() not in existing_columns
        ]

        return missing_columns

    def read(self, experiment: str) -> pd.DataFrame:
        with self._connection() as con:
            df = pd.read_sql(f"select * from {experiment};", con)
            # If a whole column only contains nan, the df will contain None
            # https://github.com/pandas-dev/pandas/issues/14314
            if None in df.values:
                df = df.replace([None], float("nan"))
            return df


class ExperimentExistsError(Exception):
    def __init__(self, experiment: str, *args, **kwargs):
        msg = (
            f"An experiment with the name '{experiment}' already exists."
            f"\n\tYou can overwrite it by setting 'overwrite=True'."
        )
        super().__init__(msg, *args, **kwargs)
