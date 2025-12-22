"""
Dag for collecting uid/uidl for reparsing based on passed filter
"""
__author__ = "artem.pereverzev"

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow.operators.sensors import WebHdfsSensor
from airflow.contrib.sensors.bash_sensor import BashSensor
from dagtools.graphite_sender import status_callbacks
from airflow.operators.python_operator import PythonOperator
from pendulum import datetime
from dagtools import strargs, sparkconf, dagdir, dsmsk

from pyspark.sql import SparkSession, functions as F
import pyspark.sql.types as T
import datetime as dt


default_args = {
    "owner": __author__,
    "start_date": datetime(2025, 1, 1),
    "start_date": datetime(2025, 1, 1),
    "retries": 3,
    **status_callbacks,
}



with DAG(
    "universal-reparse",
    doc_md=__doc__,
    default_args=default_args,
    schedule_interval=None,
) as dag:
    start = DummyOperator(task_id="start")

    find_n_publish = SparkSubmitOperator(
        task_id="find_n_publish",
        application=dagdir("find_n_publish.py"),
        name="FindAndPublsih[{{ dsmsk(execution_date) }}]",
        num_executors=3,
        executor_cores=2,
        executor_memory="1g",
        driver_memory="1g",
        conf={
            **sparkconf,
            "spark.jars": "hdfs://antispam/home/airflow-as/kafka-jars/*.jar,/usr/lib/hadoop/lib/hadoop-lzo.jar",
        },
        application_args=strargs(
            "--bootstrap-servers", "kafka1.i:9092,kafka2.i:9092,kafka3.i:9092",
            "--topic", "reparser",
            "--group-id", "reparser"
        ),
    )

    end = DummyOperator(task_id="end")

    start >> find_n_publish >> end
