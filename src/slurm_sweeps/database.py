import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

import fasteners
import pandas as pd


class DBObject(ABC):
    def __init__(self, path: Union[str, Path] = "./database"):
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

    def write(self, experiment: str, row: Dict):
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
