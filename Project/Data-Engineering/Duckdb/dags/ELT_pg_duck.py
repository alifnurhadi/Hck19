from datetime import datetime
import os
import polars as pl
from airflow import DAG
from airflow.operators.python import PythonOperator
import duckdb
import logging


Create_table = '''
CREATE TABLE IF NOT EXISTS duckduck ( 
col1 UUID ,
col2 STRING ,
col3 BIGINT ,
col4 INT ,
col5 BOOLEAN
)'''
 
coalesces = {
    'col2': 'COALESCE(col2 , ( SELECT MODE(col2) FROM duckduck )' ,
    'col3': 'COALESCE(col3 , ( SELECT MEDIAN(col3) FROM duckduck )' ,
    'col4': 'COALESCE(col4 , ( SELECT MEDIAN(col4) FROM duckduck )' ,
    'col5': 'COALESCE(col5 , FALSE) FROM duckduck' 
}

def join_query(kwargs:dict):
    combine = ', '.join([ kwargs[col] for col in kwargs.keys()])
    return f'SELECT {combine} FROM duckduck'



database_name = 'duckdb_alif.duckdb'
data_path = '/opt/airflow/data/'        # any path that file are located.

args = {
    'owner' :'ELT_duck_alif' ,
    'start_date' : datetime(2024, 10, 1) ,
    'retries':1
}

def get_data():
    '''this would be fill with anything to get the data either check into folder of get from an api or read from other databases'''
    ...

def load_into():
    '''
    this process is basicly a logic to load every data that match the spesific format into a data lake or a big database
    '''
    with duckdb.connect(database_name) as con :
        if not os.listdir(data_path):
            logging.warning("No files found in the directory.")
            return
        
        for filename in os.listdir(data_path):
            if filename.endswith('.csv'):
                # Read data using Polars
                data = pl.read_csv(os.path.join(data_path, filename)).lazy().collect(engine='gpu')
                
                # Ensure columns match the expected columns
                req_col = ['col1', 'col2', 'col3', 'col4', 'col5']
                if set(req_col).issubset(data.columns):
                    try:
                        # Create table if it doesn't exist
                        con.execute(Create_table)
                        # Insert data into DuckDB
                        con.execute('INSERT INTO duckduck SELECT * FROM data')
                    except duckdb.ConnectionException as e:
                        logging.error(f"Database connection error: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error: {e}")
                else:
                    logging.warning(f"Column mismatch in file: {filename}. Skipping.")
            else:
                logging.warning(f"Non-CSV file found in directory: {filename}. Skipping.")


def duckdb_transform():
    """
    doing handling NULL value on datas
    """
    with duckdb.connect(config={'threads': 5}, database=database_name) as con:
        try:
            combine_query = join_query(coalesces)
            con.execute(combine_query)
        except duckdb.ConnectionException as e:
            logging.error(f"Database connection error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during transformation: {e}")


with DAG('duckdb_alif',
         schedule_interval = '0 1 * * *',
         default_args = args,
         catchup = False
         ) as dag:
    

    getting_data = PythonOperator(
        task_id = 'getting a data from sources',
        python_callable=get_data,
        dag = dag
        )
    
    load_to_database = PythonOperator(
        task_id = 'load data directly into a lake or big database',
        python_callable = duckdb_transform,
        dag = dag
    )

    handle_data = PythonOperator(
        task_id = 'handle every data inside database' , 
        python_callable = duckdb_transform,
        dag = dag
    )

    getting_data >> load_to_database >> handle_data
