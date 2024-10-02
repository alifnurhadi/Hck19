from datetime import datetime
import random
import re
import string
from airflow import DAG # type: ignore
from airflow.operators.python import PythonOperator # type: ignore
import polars as pl
from sqlalchemy import create_engine, text
import os


# load env here !!!

POSTGRES_USER = "final_project"
POSTGRES_PASSWORD = "final_project"
POSTGRES_DB = "final_project"
POSTGRES_HOST = "postgres"  
POSTGRES_PORT = "5400"

db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(db_url)

schema = pl.Schema({
    'call_center':pl.Int32
})

folderinput = '/opt/airflow/data'
folderoutput = '/opt/airflow/data/clean'

number = string.digits
phone_num = ''.join(random.choice(number) for _ in range(12))

def retrieve_data():
    listdir = os.listdir(folderinput)
    cleaned_files = []

    for i, filename in enumerate(listdir, start=1):
        file_path = os.path.join(folderinput, filename)
        if filename.endswith('.csv'):
            try:
                data1 = pl.scan_csv(file_path).collect()
                cleaned_file_name = f'data_{i:02d}.csv'
                data1.write_csv(os.path.join(folderoutput, cleaned_file_name))
                cleaned_files.append(cleaned_file_name)
            except Exception as e:
                print(f'Error processing file {filename}: {e}')
    
    print(f'Terdapat data baru sebanyak {len(cleaned_files)}')
    return cleaned_files  # Return the list of cleaned files

def transform():
    listdir = os.listdir(folderinput)
    cleaned_files = []

    for i, filename in enumerate(listdir, start=1):
        if filename.endswith('.csv'):
            try:
                data = pl.read_csv(os.path.join(folderinput, filename))
                #remove duplicate
                data = data.unique(keep='first')                                                                    # remove data duplikat.
                data = data.rename({col:col.lower() for col in data.columns})                                         # menghandle ketidak seragaman nama kolom.

                # menghandle data yang null atau hilang.
                required_columns = ['kota' , 'rumah_sakit' , 'nama' , 'spesialis' , 'sub_spesialis' , 'call_center' , 'non_spesialis']

                if all(col in data.columns for col in required_columns):
                    for dat in data.columns:
                        null = data.select((pl.col(dat)).is_null()).height
                        try:
                            if null > 0 :
                                if dat == 'call_center':
                                    data = data.with_columns(pl.col(dat).fill_null(value=phone_num))

                                elif dat == re.search(r'date|tanggal', dat, re.IGNORECASE):
                                    data = data.with_columns(pl.col(dat).fill_null(strategy='forward'))

                                elif dat == 'kota':
                                    data = data.with_columns(
                                            pl.when(pl.col("kota").is_null())
                                            .then(pl.col("rumah_sakit").str.extract(r'(.+)\ (RSUD)\ (.+\w) ',2)) \
                                                .otherwise(pl.col("rumah_sakit")) \
                                                    .alias("kota")
                                        )

                                elif dat in ['sub_spesialis' , 'spesialis' ]:
                                    data = data.with_columns(pl.col(dat).fill_null(value='data tidak tersedia'))

                                elif dat == 'non_spesialis':
                                    data = data.with_columns(pl.col(dat).fill_null('-'))    

                                new_name = f'data_{i:02d}.csv'
                                final_clean_path = os.path.join(folderoutput, new_name)
                                data.write_csv(final_clean_path)
                                cleaned_files.append(new_name)
                                print(f'Data berhasil ditransformasi ke {new_name}')
                
                                print(f'data berhasil ditransformasi data ke {i}')

                                # menghapus data rawnya
                                # os.remove(file_path)
                            else:
                                continue
                        except Exception as e:
                            print(f'data already clean')
                            print(e)
                            continue
                else :
                    continue

            except Exception as e:
                print(f'Error processing file {filename}: {e}')
                    
        else:
            print(f'Skipping non-CSV file: {filename}')
            continue
    return cleaned_files     


def load_into():
    
    kumpulan = '/opt/airflow/data/clean'
    listdirr = os.listdir(kumpulan)
    csv_files = [f for f in listdirr if f.endswith('.csv')]

    if not csv_files:
        raise Exception('No CSV files found in the directory')

    for filename in csv_files:
        data = pl.read_csv(os.path.join(kumpulan, filename))
        try:
            with engine.connect() as connection:
                data.write_database(table_name='hospital_project', connection=db_url, if_table_exists='append')
        except Exception as e:
            print(f"Error writing to database with db_url: {e}")
            try:
                try:
                    data.write_database(table_name='hospital_project', connection=db_url, if_table_exists='append')
                except:
                    data.write_database(table_name='hospital_project', connection=connection, if_table_exists='append')
            except Exception as e:
                print(f"Error writing to database with db_url: {e}")
        finally:
            print(f'Successfully loaded data from {filename} into the database')
        


def_args = {
    'owner': 'Alif,Clara,Ghafar',
    'start_date': datetime(2024, 9, 1),
    'retries': 1,
}

with DAG('percobaan_1',
         schedule_interval='0 3 * * *',
         default_args=def_args,
         catchup=False) as dag:


    collect_data = PythonOperator(
        task_id = 'get',
        python_callable=retrieve_data,
        dag=dag
    )

    transforming = PythonOperator(
        task_id = 'transform',
        python_callable=transform,
        dag=dag
    )

    load_to_db = PythonOperator(
        task_id = 'load',
        python_callable=load_into,
        dag=dag
    )

    collect_data >> transforming >> load_to_db