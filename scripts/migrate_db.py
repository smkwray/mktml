import os

import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'market_data.duckdb')

con = duckdb.connect(DB_PATH)
try:
    con.execute('ALTER TABLE recommendation_history ADD COLUMN confidence DOUBLE DEFAULT 0.5')
    print('Column confidence added successfully')
except Exception as e:
    print(f'Error adding column: {e}')
finally:
    con.close()
