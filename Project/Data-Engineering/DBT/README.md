
## Overview

This repository demonstrates how **DBT (Data Build Tool)** is integrated in an **ELT (Extract, Load, Transform)** workflow orchestrated by **Apache Airflow**. In this setup, DBT is used to handle the **transformation** step of the ELT process, while Airflow manages the scheduling and automation of the pipeline.

DBT is a powerful transformation tool, which uses SQL to define and manage data transformations and dependencies. Airflow provides the orchestration layer, ensuring tasks are executed at the right time and in the correct order.

---

## Why ELT Over ETL?

In traditional ETL (Extract, Transform, Load) pipelines, the transformation happens before data loading. With the increasing demand for large-scale data processing, **ELT** has become more popular because:

1. **Performance**: ELT defers transformations until after the data is loaded into the database, allowing DBT to leverage the full power of the data warehouse or lake for transformations.
2. **Modularity**: By separating the data transformation into DBT models, each model can be easily tested, updated, and reused.
3. **Scalability**: ELT pipelines scale better because they can process raw data in its entirety and transform it later in the database.
4. **Ease of Maintenance**: DBT manages data models and dependencies, making it easier to track and update transformations as data evolves.

---

## Features

- **DBT for Transformations**: DBT is used to define SQL transformations and manage the dependencies between different data models.
- **Automated Workflow with Airflow**: Airflow automates the data pipeline, orchestrating the tasks for extracting data, loading it into the database, and running DBT transformations.
- **Error Handling**: Both DBT and Airflow come with robust logging and error handling, ensuring that any issues are properly tracked and managed.
- **Modular and Reusable**: The DBT models are reusable and can be easily tested or run independently from the entire pipeline.

---

## Installation

### Clone the Repository:
```bash
git clone https://github.com/yourusername/dbt-elt-airflow.git
```

### Install Dependencies:
- Install Python dependencies:
  ```bash
  pip install apache-airflow dbt-core
  ```
- Install DBT for your database (e.g., Postgres, Snowflake, etc.):
  ```bash
  pip install dbt-postgres   # For Postgres, for example
  ```

- Ensure that **Airflow** is properly set up in your environment and **DBT** is configured for your data warehouse.

---

## Usage

### DBT Project Setup

The DBT models are located in the `models/` directory of the DBT project. The DBT project is set up to connect to your data warehouse, where the raw data is stored.


### Airflow DAG Overview

This script defines an Airflow DAG that orchestrates the following tasks:

1. **Extract**: Retrieves data from specified sources (e.g., APIs, external databases).
2. **Load**: Loads data into the database (e.g., Postgres, BigQuery).
3. **Transform**: Triggers DBT to run data transformations on the loaded data.

---

### Airflow Python and Bash Operators

- **PythonOperator** is used for the **Extract** and **Load** steps to manage Python logic directly.
- **BashOperator** is used to run DBT commands, such as `dbt run` to trigger the transformation models defined in DBT.

---

## Future Improvements

- **Data Validation**: You can add DBT tests as part of the Airflow DAG using `dbt test` to validate the transformed data before further consumption.
- **Incremental Loading**: Implement incremental data loading to optimize performance for large datasets.
- **Parallel DBT Runs**: If your DBT transformations can run in parallel, adjust the DAG configuration to enable parallel execution.
  
---

## License

This project is open-source and free to use under the MIT License.

---

## Author

Created by alif nurhadi, 2024.
