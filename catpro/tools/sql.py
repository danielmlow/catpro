import pandas as pd
import mysql.connector as sql
import numpy as np
import time

def query_to_df(q, host,
                port, database,
                user, password,print_timelapse = True):
	start = time.time()
	db_connection = sql.connect(host=host,
	                            port=port, database=database,
	                            user=user, password=password)
	db_cursor = db_connection.cursor(buffered=True)
	db_cursor.execute(q)
	table_rows = db_cursor.fetchall()

	df = pd.DataFrame(table_rows, columns  = db_cursor.column_names)
	if print_timelapse:
		seconds = time.time() - start
		print(f'query took {np.round(seconds,3)} sec or {np.round(seconds/60,3)} min')
	return df