import datetime
import sqlite3 as sl


def generateLogQuery(table_name, trial_id, iteration, user_key, user_value):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    user_log = f"{user_key}: {user_value}"

    return f"""INSERT INTO {table_name} (trial_id, timestamp, iteration, logged_by_user)
    VALUES ('{trial_id}', '{timestamp}', {iteration}, '{user_log}');"""


def generateCreateQuery(table_name):
    return (
        """
            CREATE TABLE """
        + table_name
        + """ (
                trial_id        TEXT,
                timestamp       TEXT,
                iteration       NUMERIC,
                logged_by_user  TEXT
            );
        """
    )


def fetchTableNames(path):
    with DBConnection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT name FROM sqlite_master
                         WHERE type = 'table';"""
        )
        return cursor.fetchall()


class DBConnection(object):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.conn = sl.connect(self.path, isolation_level=None)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


class SSDB:
    def experiment_exists(path, table_name):
        return (table_name,) in fetchTableNames(path)

    def create_table(path, table_name):
        with DBConnection(path) as conn:
            query = generateCreateQuery(table_name)
            conn.execute(query)

    def insert_log(path, table_name, trial_id, iteration, user_key, user_value):
        with DBConnection(path) as conn:
            query = generateLogQuery(
                table_name, trial_id, iteration, user_key, user_value
            )
            conn.execute(query)

    def reset_table(path, table_name):
        with DBConnection(path) as conn:
            conn.execute("DROP TABLE IF EXISTS " + table_name + ";")
            query = generateCreateQuery(table_name)
            conn.execute(query)
