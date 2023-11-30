import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cloudpickle
import pandas as pd
import yaml

from .constants import (
    DB_CFG,
    DB_END_TIME,
    DB_EXPERIMENT,
    DB_ITERATION,
    DB_LOGGED,
    DB_METRIC,
    DB_METRICS,
    DB_OBJECT_DATA,
    DB_OBJECT_NAME,
    DB_START_TIME,
    DB_STATUS,
    DB_STORAGE,
    DB_TIMESTAMP,
    DB_TRIAL_ID,
    DB_TRIALS,
)
from .trial import Trial


class SqlDatabase:
    """An SQLite database that stores the trials of an experiment and their metrics.

    It also serves as a storage for pickled objects.

    Args:
        experiment: Name of the experiment.
        path: The path to the database file.
    """

    def __init__(self, experiment: str, path: Union[str, Path] = "./slurm_sweeps.db"):
        self._experiment = experiment
        self._path = Path(path).resolve()

        if not self.path.exists():
            with self._connection() as con:
                con.execute(
                    f"""
                    create table {DB_STORAGE} (
                        {DB_TIMESTAMP} datetime default(strftime('%Y-%m-%d %H:%M:%f', 'NOW')),
                        {DB_EXPERIMENT} text not null,
                        {DB_OBJECT_NAME} text not null,
                        {DB_OBJECT_DATA} blob,
                        unique({DB_EXPERIMENT}, {DB_OBJECT_NAME})
                    );"""
                )

    @property
    def experiment(self) -> str:
        """The name of the experiment."""
        return self._experiment

    @property
    def path(self) -> Path:
        """The resolved absolute path to the database file."""
        return self._path.resolve()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            yield connection
        finally:
            connection.close()

    def exists(self) -> bool:
        """Check if the experiment exists in the database."""
        with self._connection() as con:
            response = con.execute(
                f"select name from sqlite_master where type='table' "
                f"and name in ('{self.experiment}{DB_TRIALS}', '{self.experiment}{DB_METRICS}')"
            ).fetchall()

        return len(response) == 2

    def create(self, overwrite: bool = False):
        """Create the trials and metrics table for the experiment.

        Args:
            overwrite: Overwrite tables if they already exist.

        Raises:
            ExperimentExistsError if `overwrite=False` and tables already exist.
        """
        with self._connection() as con:
            if overwrite:
                con.execute(f"drop table if exists {self.experiment}{DB_TRIALS};")
                con.execute(f"drop table if exists {self.experiment}{DB_METRICS};")

            try:
                self._create_trials_table(con)
                self._create_metrics_table(con)
            except sqlite3.OperationalError as err:
                if "already exists" in str(err):
                    raise ExperimentExistsError(self.experiment) from err
                raise err

    def _create_trials_table(self, con: sqlite3.Connection):
        """Helper method to create the trials table."""
        con.execute(
            f"""
            create table {self.experiment}{DB_TRIALS}(
                {DB_TIMESTAMP} datetime default(strftime('%Y-%m-%d %H:%M:%f', 'NOW')),
                {DB_TRIAL_ID} text primary key,
                {DB_CFG} text,
                {DB_START_TIME} datetime,
                {DB_END_TIME} datetime,
                {DB_STATUS} text
            );"""
        )

    def _create_metrics_table(self, con: sqlite3.Connection):
        """Helper method to create the metrics table."""
        con.execute(
            f"""
            create table {self.experiment}{DB_METRICS}(
                {DB_TIMESTAMP} datetime default(strftime('%Y-%m-%d %H:%M:%f', 'NOW')),
                {DB_TRIAL_ID} text,
                {DB_ITERATION} integer
            );"""
        )

    def write_trial(self, trial: Trial):
        """Add a trial to the experiment trials table."""
        start_time = str(trial.start_time) if trial.start_time else trial.start_time
        end_time = str(trial.end_time) if trial.end_time else trial.end_time
        status = trial.status.value if trial.status else trial.status

        with self._connection() as con:
            con.execute(
                f"""
                insert or replace into {self.experiment}{DB_TRIALS}(
                    {DB_TRIAL_ID}, {DB_CFG}, {DB_START_TIME}, {DB_END_TIME}, {DB_STATUS}
                ) values (?, ?, ?, ?, ?);
                """,
                (
                    trial.trial_id,
                    yaml.safe_dump(trial.cfg, None),
                    start_time,
                    end_time,
                    status,
                ),
            )

    def read_trials(self, trial_id: Optional[str] = None) -> List[Trial]:
        """Read the trials table and return a list of Trials."""
        where = ""
        if trial_id:
            where = f"where {DB_TRIAL_ID} = '{trial_id}'"

        with self._connection() as con:
            response = con.execute(
                f"""
                select * from {self.experiment}{DB_TRIALS} {where};
                """
            ).fetchall()

        trials = [
            Trial(
                cfg=yaml.safe_load(row[2]),
                start_time=datetime.fromisoformat(row[3]) if row[3] else row[3],
                end_time=datetime.fromisoformat(row[4]) if row[4] else row[4],
                status=row[5],
            )
            for row in response
        ]

        return trials

    def write_metrics(
        self, trial_id: str, iteration: int, metrics: Dict[str, Union[int, float]]
    ):
        """Add some metrics to the experiment metrics table.

        Args:
            trial_id: The ID of the trial.
            iteration: The iteration of the metrics.
            metrics: The metrics to be written.
        """
        if not metrics:
            return

        metrics = self._add_prefix_and_logged_flag(metrics)
        table = f"{self.experiment}{DB_METRICS}"
        with self._connection() as con:
            # add missing columns
            columns_and_types = self._get_missing_metric_columns_and_types(con, metrics)
            for col, typ in columns_and_types:
                try:
                    con.execute(f"alter table {table} add column {col} {typ}")
                # Just as a safeguard, during tests I sometimes encountered this error ...
                except sqlite3.OperationalError as err:
                    if "no such table" in str(err):
                        raise ExperimentNotFoundError(self.experiment)
                    elif "duplicate column name" in str(err):
                        pass
                    else:
                        raise err

            # insert metrics
            con.execute(
                f"insert into {table}({DB_TRIAL_ID}, {DB_ITERATION}, {', '.join(metrics.keys())}) "
                f"values(?, ?, {', '.join(['?' for _ in range(len(metrics))])});",
                [trial_id, iteration] + list(metrics.values()),
            )

    @staticmethod
    def _add_prefix_and_logged_flag(
        metrics: Dict[str, Union[int, float]]
    ) -> Dict[str, Union[int, float]]:
        """Helper method to change the metrics into the right format."""
        return {
            k: v
            for key, value in metrics.items()
            for k, v in zip(
                (f"{DB_METRIC}{key}", f"{DB_METRIC}{key}{DB_LOGGED}"), (value, 1)
            )
        }

    def _get_missing_metric_columns_and_types(
        self, connection: sqlite3.Connection, metrics: Dict
    ) -> List[Tuple[str, str]]:
        """Helper method to get the missing metric columns and their sql types."""

        def get_sql_type(value: Union[float, int]) -> str:
            if type(value) is int:
                return "integer"
            if type(value) is float:
                return "real"
            return "numeric"

        response = connection.execute(
            f"pragma table_info({self.experiment}{DB_METRICS})"
        ).fetchall()
        existing_columns = [col[1].lower() for col in response]
        missing_columns_and_types = [
            (col.lower(), get_sql_type(val))
            for col, val in metrics.items()
            if col.lower() not in existing_columns
        ]

        return missing_columns_and_types

    def read_metrics(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Read all metrics or one specific metric from the experiment_metrics table.

        Args:
            metric: When specified, return only this metric and filter for its logged flag.

        Returns:
            The metrics table as a pandas DataFrame.
        """
        columns, where = "*", ""
        if metric:
            columns = f"{DB_TRIAL_ID}, {DB_ITERATION}, {DB_METRIC}{metric}"
            where = f"where {DB_METRIC}{metric}{DB_LOGGED} = 1"

        with self._connection() as con:
            df = pd.read_sql(
                f"select {columns} from {self.experiment}{DB_METRICS} {where};", con
            )
            # If a whole column only contains nan, the df will contain None
            # https://github.com/pandas-dev/pandas/issues/14314
            if None in df.values:
                df = df.replace([None], float("nan"))
            return df

    def dump(self, data: Dict[str, Any]):
        """Pickles and dumps the data to the storage table.

        Args:
            data: A dictionary with the names and the corresponding objects to be pickled.
        """
        data = [
            (self.experiment, name, cloudpickle.dumps(obj))
            for name, obj in data.items()
        ]
        with self._connection() as con:
            con.executemany(
                f"insert or replace into {DB_STORAGE}({DB_EXPERIMENT}, {DB_OBJECT_NAME}, {DB_OBJECT_DATA}) "
                f"values(?, ?, ?);",
                data,
            )

    def load(self, name: str) -> Any:
        """Loads the pickled object from the storage table.

        Args:
            name: The name of the pickled object in the storage table.

        Returns:
            The unpickled object.
        """
        with self._connection() as con:
            response = con.execute(
                f"select {DB_OBJECT_DATA} from {DB_STORAGE} "
                f"where {DB_EXPERIMENT}='{self.experiment}' and {DB_OBJECT_NAME}='{name}'"
            ).fetchone()

        return cloudpickle.loads(response[0])


class ExperimentNotFoundError(Exception):
    def __init__(self, experiment: str, *args, **kwargs):
        msg = (
            f"An experiment with the name '{experiment}' was not found in the database."
            f"\n\tYou can create one by calling `SqlDatabase(...).create()` "
            f"or by setting `Experiment(..., restore=False)`."
        )
        super().__init__(msg, *args, **kwargs)


class ExperimentExistsError(Exception):
    def __init__(self, experiment: str, *args, **kwargs):
        msg = (
            f"An experiment with the name '{experiment}' already exists."
            f"\n\tYou can overwrite it by setting 'overwrite=True'."
        )
        super().__init__(msg, *args, **kwargs)
