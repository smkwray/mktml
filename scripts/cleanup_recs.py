import os

import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'market_data.duckdb')

con = duckdb.connect(DB_PATH)
try:
    con.execute('DROP TABLE IF EXISTS recommendation_history')
    print('Table recommendation_history dropped')
finally:
    con.close()
