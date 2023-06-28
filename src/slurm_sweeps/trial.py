import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Status(Enum):
    """A trial status"""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of"
            f" {list(cls._value2member_map_.keys())}"
        )


@dataclass
class Trial:
    cfg: Dict
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def trial_id(self) -> str:
        json_bytes = json.dumps(self.cfg, sort_keys=True).encode()
        return hashlib.blake2s(json_bytes, digest_size=6).hexdigest()

    @property
    def status(self) -> Status:
        """Return the status of the trial."""
        if self.process is None:
            return Status.SCHEDULED
        if self.process.poll() is None:
            return Status.RUNNING
        if self.process.poll() == 0:
            return Status.COMPLETED
        return Status.PRUNED

    @property
    def terminated(self) -> bool:
        """Return True, if the trial has been completed or pruned."""
        return self.status in [Status.COMPLETED, Status.PRUNED]

    @property
    def runtime(self) -> Optional[float]:
        """Return the runtime of the trial."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        if self.start_time is not None:
            return time.time() - self.start_time
        return None
