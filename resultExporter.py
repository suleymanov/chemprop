import pandas as pd
import sqlite3 as db


engine = db.connect('/extra/dwicke/chemprop/mymodels/regression/results.db')
sql_df = pd.read_sql('SELECT * FROM results', engine)

sql_df.to_excel('~/results.xlsx', index=False)
