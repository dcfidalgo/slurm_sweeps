import datetime
import json
import sqlite3 as sl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

import fasteners
import pandas as pd


class DBObject(ABC):
    def __init__(self, path: Union[str, Path]):
        self._path = Path(path).resolve()

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass


class FileDatabase(DBObject):
    def __init__(self, path):
        super().__init__(path)
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _get_file_path_and_lock(
        self, experiment
    ) -> Tuple[Path, fasteners.InterProcessReaderWriterLock]:
        path = self.path / f"{experiment}.txt"
        lock = fasteners.InterProcessReaderWriterLock(f"{path}.lock")

        return path, lock

    def create(self, experiment: str, exist_ok: bool = False):
        path, _ = self._get_file_path_and_lock(experiment)
        path.touch(exist_ok=exist_ok)
        with path.open(mode="w"):
            pass

    def write(self, experiment: str, row: Dict, key: str = None, value=None):
        path, lock = self._get_file_path_and_lock(experiment)

        json_str = json.dumps(row, sort_keys=True)

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


class DBConnection(object):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.conn = sl.connect(self.path, isolation_level=None)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


class SQLDatabase(DBObject):
    def __init__(self, path):
        super().__init__(path)

    @property
    def path(self) -> Path:
        return self._path

    def generateCreateQuery(self, table_name):
        return f"""CREATE TABLE {table_name} (
                    trial_id        TEXT,
                    timestamp       TEXT,
                    iteration       NUMERIC,
                    logged_by_user  TEXT
                );
            """

    def create(self, experiment: str, exist_ok: bool = False):
        with DBConnection(self.path) as conn:
            if exist_ok:
                conn.execute(f"DROP TABLE IF EXISTS {experiment};")

            query = self.generateCreateQuery(experiment)
            conn.execute(query)

    def read(self, experiment: str) -> pd.DataFrame:
        with DBConnection(self.path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {experiment};", conn)

    def write(self, experiment: str, data: Dict, key: str = None, value=None):
        with DBConnection(self.path) as conn:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            trial_id = data["trial_id"]
            iteration = data["iteration"]
            user_log = json.dumps({key: value})

            conn.execute(
                f"""INSERT INTO {experiment}
                (trial_id, timestamp, iteration, logged_by_user)
                VALUES (?, ?, ?, ?);""",
                (trial_id, timestamp, iteration, user_log),
            )
