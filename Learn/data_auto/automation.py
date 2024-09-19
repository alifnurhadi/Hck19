from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import polars as pl
from sqlalchemy import create_engine
from elasticsearch import Elasticsearch

'''
Mendefinisikan / menginisialisasi beberapa variabel untuk tools seperti database dan kebutuhan visualiasi

NOTES : it's not recommend to hardcode variabel tersebut dalam sebuah script, 
(opsi membuat file .env dan menaruhnya di .gitignore atau membuat semacam secret file yang bisa diakses dari file scripts ini.)
'''

POSTGRES_USER = "milestone_3rd"
POSTGRES_PASSWORD = "milestone_3rd"
POSTGRES_DB = "milestone_3rd"
POSTGRES_HOST = "postgres"  
POSTGRES_PORT = "5432"

ELASTICSEARCH_HOST = "elasticsearch" 
ELASTICSEARCH_PORT = "9200"

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
ELASTICSEARCH_URL = f"http://{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}"

def load_csv_to_postgres():
    '''
    function ini dibuat untuk melakukan task mmasukkan data kedalam database postgres.
    '''

    # Create the engine and connect to the PostgreSQL database
    engine = create_engine(POSTGRES_URL)
    conn = engine.connect()

    # Load CSV with Polars
    try :
        df = pl.read_csv('/opt/airflow/data/Dataset_milestone.csv')
    except :
        df = pl.read_csv('../data/Dataset_milestone.csv')

    # Write Polars DataFrame directly to PostgreSQL
    df.write_database(table_name='milestone' ,connection=POSTGRES_URL, if_table_exists='replace')
    
    # logging hasil data yang di input
    print(f"Loaded {len(df)} rows into PostgreSQL")

def get_data():
    '''
    function ini dibuat untuk melakukan pengambilan data dari sebuah database.
    '''

    # Create the engine and connect to the PostgreSQL database
    engine = create_engine(POSTGRES_URL)
    conn = engine.connect()

    # Execute SQL query and convert result to Polars DataFrame
    df = pl.read_database(query="SELECT * FROM milestone", connection=conn)

    # Save result as CSV
    df.write_csv('/opt/airflow/data/P2M3_alif_nurhadi_raw.csv')

    # logging hasil data yang dibaca.
    print(f"Saved {len(df)} rows to P2M3_alif_nurhadi_raw.csv")


def transform():
    '''
    function ini dibuat untuk menghandle data yang duplikat dan data yang tidak tersisi.
    '''

    engine = create_engine(POSTGRES_URL)                                        # Create the engine and connect to the PostgreSQL database
    conn = engine.connect()

    # membaca data yang ada di database
    df = pl.read_database(query="SELECT * FROM milestone",connection= conn)
    df = df.unique(keep='first')                                                                    # remove data duplikat.
    df = df.rename({col:col.lower() for col in df.columns})                                         # menghandle ketidak seragaman nama kolom.
    
    # menghandle data yang null atau hilang.
    for data in df.columns:
        null = df.select(pl.col(data)).is_empty()
        if null > 0 :
            if df[data].dtype  == pl.Utf8 :
                df = df.with_columns(pl.col(data).fill_null(strategy='backward'))
            elif df[data].dtype in [pl.Float64, pl.Float32, pl.Int32, pl.Int64 ] :
                df = df.with_columns(pl.col(data).fill_null().median())
            elif data == 'date' or 'Date':
                df = df.with_columns([pl.col('date').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S',strict=False)])
                df = df.with_columns(pl.col(data).fill_null(strategy='backward'))    
        else:
            continue
    
    # men-ekstrak data yang sudah dibersihkan dari nilai null , duplikat, dan ketidak seragaman data
    df.write_csv('/opt/airflow/data/P2M3_alif_nurhadi_clean.csv')
    print(f"Transformed and saved {len(df)} rows to P2M3_alif_nurhadi_clean.csv")                   # logging hasil data yang dibaca.


def load_to_elastic():
    try:
        es = Elasticsearch(hosts=["http://elasticsearch:9200"])                             # mengkoneksikan dengan elastic.

        if not es.ping():
            print("Elasticsearch server is not reachable")                                  # menghandle lebih awal jika gagal connect.
            return

        # Load cleaned data with Polars
        df = pl.read_csv('/opt/airflow/data/P2M3_alif_nurhadi_clean.csv')
        
        print(f"Total rows in DataFrame: {len(df)}")

        successful_indexing = 0 
        for i, row in enumerate(df.iter_rows(named=True)):                         # melakukan iterasi pada data untuk menghasilkan unique identifier
            doc = dict(row)
            # create exception handling jika proses load ke elastic gagal
            try:
                res = es.index(index="milestone", id=i+1, body=doc)                # Use the unique ID here
                print(f"Indexed document {i} in Elasticsearch: {res}")
                successful_indexing += 1
            except Exception as e:
                print(f"Error indexing document {i} with data {doc}: {e}")
                continue  # Skip this document and continue with the next
        
        print(f"Total documents successfully indexed: {successful_indexing}")      # logging purpose

        es.indices.refresh(index="milestone")                                      # langkah logging dari hasil inputisasi ke elastic.
        doc_count = es.count(index="milestone")['count']
        print(f"Total documents in Elasticsearch index: {doc_count}")

    except ConnectionError:
        print("Failed to connect to Elasticsearch")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# men-define variabel default untuk proses orchestration.
default_args = {
    'owner': 'hck_milestone_3',
    'start_date': datetime(2024, 5, 1),
    'retries': 1,
}

# men-define variabel dag yang berisi how the airflow works.
with DAG("Project_Milestone_alif",
         schedule_interval='30 6 * * *',
         default_args=default_args,
         catchup=False) as dag :

    # set task 1
    load_csv_task = PythonOperator(
        task_id='load_csv_to_postgres',
        python_callable=load_csv_to_postgres,
        dag=dag) 

    # set task 2
    fetch_data = PythonOperator(
        task_id='fetch_data',
        python_callable=get_data,
        dag=dag)

    # set task 3
    transform_data = PythonOperator(
        task_id='clean_up',
        python_callable=transform,
        dag=dag)

    # set task 4
    upload_data = PythonOperator(
        task_id='upload_data_elastic',
        python_callable=load_to_elastic,
        dag=dag)

    # mengkonstruksi struktur airflow.
    load_csv_task >> fetch_data >> transform_data >> upload_data
