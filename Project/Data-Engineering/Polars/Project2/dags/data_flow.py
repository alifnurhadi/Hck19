from datetime import datetime
import random
import string
from airflow import DAG
from airflow.operators.python import PythonOperator
import polars as pl
from sqlalchemy import create_engine

# Load environment variables here if necessary

POSTGRES_USER = "final_project"
POSTGRES_PASSWORD = "final_project"
POSTGRES_DB = "final_project"
POSTGRES_HOST = "postgres"
POSTGRES_PORT = "5400"

db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(db_url)

name = 'raw'
path = f'/opt/airflow/data/{name}'
phone_num = ''.join(random.choice(string.digits) for _ in range(12))


def retrieve_data():
    data1 = pl.read_csv('/opt/airflow/data/rsha_bandung_2.csv')
    data1.write_csv(f'{path}.csv')
    print(f'Finished collecting {len(data1)} records')


def transform():
    data = pl.read_csv(f'{path}.csv')
    data = data.unique(keep='first').rename({col: col.lower() for col in data.columns})

    # Handle null values
    for col in data.columns:
        if data[col].is_empty().all():
            continue

        if col == 'call_center':
            data = data.with_columns(pl.col(col).fill_null(value=phone_num))
        elif col == 'kota':
            data = data.with_columns(
                pl.when(pl.col("kota").is_null())
                .then(pl.col("rumah_sakit").str.extract(r'(.+)\ (RSUD)\ (.+\w)', 3))
                .otherwise(pl.col("rumah_sakit"))
                .alias("kota")
            )
        elif col in ['sub_spesialis', 'spesialis']:
            data = data.with_columns([
                pl.when(pl.col("spesialis").is_null()).then(pl.col("sub_spesialis")).otherwise(pl.col("spesialis")).alias("spesialis"),
                pl.when(pl.col("sub_spesialis").is_null()).then(pl.col("spesialis")).otherwise(pl.col("sub_spesialis")).alias("sub_spesialis")
            ])
        elif col in ['date', 'Date']:
            data = data.with_columns(pl.col('date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S', strict=False))
            data = data.with_columns(pl.col(col).fill_null(strategy='backward'))

    data = data.with_columns([pl.lit(None).alias('no'), pl.lit(None).alias('rating')])\
        .select([pl.col('no'), pl.exclude(['no', 'rating']), pl.col('rating')])

    data.write_csv(f'{path}_clean.csv')
    print(f'Data has been cleaned, total rows: {len(data)}')


def load_into():
    cleaned_file_count = 1  # Change this to the correct number of cleaned files
    for dats in range(cleaned_file_count):
        data = pl.read_csv(f'{path}_{dats}_clean.csv')
        try:
            with engine.connect() as connection:
                data.write_database(table_name='hospital_project', connection=connection, if_table_exists='append')
            print(f'Successfully inputting data_{dats} into the database')
        except Exception as e:
            print(f"Error writing to database: {e}")


def_args = {
    'owner': 'Alif, Clara, Ghafar',
    'start_date': datetime(2024, 9, 1),
    'retries': 1,
}

with DAG("dok_dimana_dok",
         schedule_interval='@monthly',
         default_args=def_args,
         catchup=False) as dag:

    collect_data = PythonOperator(
        task_id='task1',
        python_callable=retrieve_data,
        dag=dag
    )

    transforming = PythonOperator(
        task_id='task2',
        python_callable=transform,
        dag=dag
    )

    load_to_db = PythonOperator(
        task_id='task3',
        python_callable=load_into,
        dag=dag
    )

    collect_data >> transforming >> load_to_db
