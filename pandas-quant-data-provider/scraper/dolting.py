import os
import time
import logging
import pandas as pd
from doltpy.core import Dolt
from doltpy.etl import get_df_table_writer

REPO_LOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantdb")
_log = logging.getLogger(__name__)


def import_df(table, df, repo=REPO_LOC, **kwargs):
    get_df_table_writer(table, lambda: df, **kwargs)(repo)


def start_server_and_get_engine(repo=REPO_LOC):
    repo = Dolt(repo)
    repo.sql_server()
    time.sleep(2)

    engine = repo.get_engine()
    _log.info("\n" + str(pd.read_sql("show tables;", engine)))
    return engine


def mysql_replace_into(table, conn, keys, data_iter):
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql.expression import Insert

    @compiles(Insert)
    def replace_string(insert, compiler, **kw):
        s = compiler.visit_insert(insert, **kw)
        s = s.replace("INSERT INTO", "REPLACE INTO")
        return s

    data = [dict(zip(keys, row)) for row in data_iter]

    conn.execute(table.table.insert(replace_string=""), data)


def dolt_commit(table, message, repo=REPO_LOC):
    repo = Dolt(repo)
    repo.add(table)
    repo.commit(message)


if __name__ == '__main__':
    repo = Dolt(REPO_LOC)

    try:
        repo.sql_server()
        time.sleep(2)

        engine = repo.get_engine()
        df = pd.read_sql("show tables;", engine)
        print(df)
    finally:
        repo.sql_server_stop()
        pass

    # after data is loaded
    # repo.add('great_players')
    # repo.commit('Added some great players')
    # repo.push('origin', 'master')