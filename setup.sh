curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.1.0/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins ./config
docker compose run airflow-cli airflow config list 
sudo chmod -R 777 ./config
