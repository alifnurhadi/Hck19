#!/bin/bash
echo "Setting up environment..."
export AIRFLOW_HOME=~/airflow
airflow db init
pip install -r ../requirements.txt
