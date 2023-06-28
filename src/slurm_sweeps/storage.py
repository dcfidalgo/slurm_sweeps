from enum import Enum
from pathlib import Path
from typing import Any, Union

import cloudpickle
import yaml


class FileExtension(Enum):
    """A file extension"""

    PKL = ".pkl"
    YML = ".yml"
    PY = ".py"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of"
            f" {list(cls._value2member_map_.keys())}"
        )


class Storage:
    """Class to store/retrieve objects for an experiment."""

    def __init__(self, storage_path: Union[Path, str]):
        self._storage_path = Path(storage_path)

    @property
    def path(self) -> Path:
        """Absolute storage path"""
        return self._storage_path.resolve()

    def dump(self, obj: Any, file: Union[str, Path]) -> Path:
        """Dump object in the storage.

        Args:
            obj: Object to store.
            file: Store object in this file.

        Returns:
            Absolute path to file.
        """
        file_path = self.path / Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        ext = FileExtension(file_path.suffix)
        if ext == FileExtension.PKL:
            self._dump_pickle(obj, file_path)
        elif ext == FileExtension.YML:
            assert isinstance(obj, dict)
            self._dump_yaml(obj, file_path)
        elif ext == FileExtension.PY:
            assert isinstance(obj, str)
            self._dump_file(obj, file_path)
        else:
            raise NotImplementedError(
                f"Dumping '{ext.value}' files is not implemented."
            )

        return file_path.resolve()

    @staticmethod
    def _dump_pickle(obj: Any, path: Path):
        with path.open(mode="wb") as f:
            f.write(cloudpickle.dumps(obj))

    @staticmethod
    def _dump_yaml(cfg: dict, path: Path):
        with path.open(mode="w") as file:
            yaml.safe_dump(cfg, file)

    @staticmethod
    def _dump_file(string: str, path: Path):
        with path.open(mode="w") as file:
            file.write(string)

    def load(self, file: Union[str, Path]) -> Any:
        file_path = self.path / Path(file)
        ext = FileExtension(file_path.suffix)
        if ext == FileExtension.PKL:
            obj = self._load_pickle(file_path)
        elif ext == FileExtension.YML:
            obj = self._load_yaml(file_path)
        elif ext == FileExtension.PY:
            obj = self._load_file(file_path)
        else:
            raise NotImplementedError(
                f"Loading '{ext.value}' files is not implemented."
            )

        return obj

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with path.open("rb") as file:
            obj = cloudpickle.loads(file.read())
        return obj

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r") as file:
            dictionary = yaml.safe_load(file.read())
        return dictionary

    @staticmethod
    def _load_file(path: Path) -> str:
        with path.open() as file:
            string = file.read()
        return string

    def get_path(self, file: Union[str, Path]) -> Path:
        """Get absolute path of the file in the storage.

        Raises:
            FileNotFoundError: If the file does not exist in the storage.
        """
        file_path = self.path / file
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist!")
        return file_path.resolve()
