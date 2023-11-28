import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Union


class Status(Enum):
    """A trial status"""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}, please select one of"
            f" {list(cls._value2member_map_.keys())}"
        )


@dataclass
class Trial:
    """A trial of an experiment.

    Args:
        cfg: The config of the trial.
        process: The subprocess that runs the trial.
        start_time: The start time of the trial.
        end_time: The end time of the trial.
        status: Status of the trial. If `process` is not None, we will always query the process for the status.
    """

    cfg: Dict
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[Union[str, Status]] = None
    _status: Optional[Status] = field(init=False, repr=False)

    @property
    def trial_id(self) -> str:
        """The trial ID is a 6-digit hash from the config."""
        json_bytes = json.dumps(self.cfg, sort_keys=True).encode()
        return hashlib.blake2s(json_bytes, digest_size=6).hexdigest()

    @property
    def status(self) -> Optional[Status]:
        """The status of the trial."""
        if self.process is None:
            return self._status
        if self.process.poll() is None:
            self._status = Status.RUNNING
        elif self.process.poll() == 0:
            self._status = Status.COMPLETED
        else:
            self._status = Status.PRUNED

        return self._status

    @status.setter
    def status(self, status: Optional[Union[str, Status]]):
        if type(status) is property or status is None:
            self._status = None
        else:
            self._status = Status(status)

    @property
    def runtime(self) -> Optional[timedelta]:
        """The runtime of the trial."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        if self.start_time is not None:
            return datetime.now() - self.start_time
        return None

    def is_terminated(self) -> bool:
        """Return True, if the trial has been completed or pruned."""
        return self.status in [Status.COMPLETED, Status.PRUNED]
