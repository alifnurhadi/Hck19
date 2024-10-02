

## Overview

This repository demonstrates how **DuckDB** is used in an **ELT (Extract, Load, Transform)** environment with **Apache Airflow** as the orchestration tool. The setup utilizes DuckDB for data storage and transformation, leveraging its high-performance analytics capabilities. The script automates the process of loading CSV data into DuckDB, handling missing values using SQL functions like `COALESCE`, and transforming the data to ensure data consistency.

---

## Why ELT Over ETL?

In traditional ETL (Extract, Transform, Load) pipelines, data transformations happen before loading into the database. However, ELT is increasingly favored for large-scale data processing because:

1. **Performance**: ELT defers the transformation step until after loading the raw data into the database. DuckDB excels here, allowing fast in-database transformations.
2. **Scalability**: ELT scales better by separating concerns. Data is stored first, and transformations happen afterward in a database optimized for querying.
3. **Flexibility**: Transformations can be run as needed without needing to extract and reload data, saving time and computational resources.
4. **Cost Efficiency**: Offloading complex transformations to the database engine reduces the load on external tools and speeds up processes.

DuckDB is perfectly suited for ELT environments as it provides efficient data handling for large datasets directly on local storage, without requiring a distributed setup like traditional data warehouses.

---

## Features

- **DuckDB as the Database**: DuckDB is a fast, embeddable database system, ideal for in-process data management.
- **Automated Workflow with Airflow**: This script uses Airflow's `PythonOperator` to automate the data pipeline:
  - **Extract**: Retrieves data from specified sources (CSV files).
  - **Load**: Loads the data into DuckDB for processing.
  - **Transform**: Performs data transformations, such as handling missing values using DuckDB SQL functions like `COALESCE`, `MEDIAN`, and `MODE`.
- **Error Handling**: Logs and handles various errors during database operations, ensuring reliability and robustness.
- **Configurable Number of Threads**: The script allows multi-threaded DuckDB operations, making it highly performant for processing large datasets.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/duckdb-elt-airflow.git
   ```

2. **Install Dependencies**:
   - Install Python dependencies:
     ```bash
     pip install polars duckdb airflow
     ```
   - Ensure that Airflow is properly set up in your environment.

3. **Set Up Airflow**:
   Ensure Airflow is configured and running with the appropriate `DAGs` directory pointing to this script.

---

## Usage

### Airflow DAG Overview

The script defines an Airflow DAG that automates the following tasks:

1. **`get_data()`**: This function is where you define how to extract data (from APIs, databases, etc.). In this example, it's expected to scan the `data_path` for CSV files.
   
2. **`load_into()`**: This function loads data into DuckDB. It scans for CSV files in the specified `data_path`, reads them using Polars, and inserts them into DuckDB.

3. **`duckdb_transform()`**: This function performs transformations on the loaded data, specifically handling NULL values by applying SQL functions like `COALESCE`, `MEDIAN`, and `MODE`.


### Airflow DAG Workflow

- **Getting Data**: Simulates the process of retrieving data from a source.
- **Loading Data**: Reads CSV files and loads them into DuckDB.
- **Handling Missing Data**: Uses DuckDB SQL functions to handle NULL values in the loaded dataset.

---

## Error Handling

The script has built-in error handling:
- **Connection Errors**: Captures DuckDB connection issues and logs them.
- **Data Mismatch**: If a CSV file doesn't contain the required columns, the script logs a warning and skips the file.
- **Invalid CSV Files**: Non-CSV files are also skipped with a warning.

---

## Future Improvements

- **Dynamic Data Sources**: Enhance the `get_data()` function to connect to external APIs or databases.
- **Enhanced Transformations**: Add more sophisticated data transformations, such as data normalization and enrichment.
- **Testing and CI**: Implement continuous integration to automate testing of the workflow.

---

## License

This project is open-source and free to use under the MIT License.

---

## Author

Created by alif nurhadi, 2024.
