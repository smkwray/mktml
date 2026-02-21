import duckdb
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
path = os.path.join(PROJECT_ROOT, 'data', 'market_data.duckdb')
if not os.path.exists(path):
    print(f'DB Missing at {path}')
else:
    con = duckdb.connect(path)
    print('Tables:', con.execute('SHOW TABLES').fetchall())
    print('Total Price Rows:', con.execute('SELECT COUNT(*) FROM price_history').fetchone()[0])
    print('Total Uniq Tickers:', con.execute('SELECT COUNT(DISTINCT ticker) FROM price_history').fetchone()[0])
    
    # Check Recommendations Schema and Content
    print('--- Recommendation History ---')
    print('Schema:', con.execute('DESCRIBE recommendation_history').fetchall())
    recs_count = con.execute('SELECT COUNT(*) FROM recommendation_history').fetchone()[0]
    print(f'Total Recommendations: {recs_count}')
    if recs_count > 0:
        sample = con.execute('SELECT * FROM recommendation_history LIMIT 5').fetchall()
        print(f'Sample Data: {sample}')
    
    con.close()
