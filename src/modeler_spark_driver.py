from jobs.prophet_modeler import ProphetModeler
from pyspark.sql import SparkSession
import sys
import yaml

if __name__ == '__main__':
    spark_session = SparkSession.builder.appName('TimeSeriesForecastModeler').getOrCreate()

    if len(sys.argv) != 2:
        print("arg1 must be the config YAML")
        exit(1)

    with open(sys.argv[1]) as file:
        config = yaml.safe_load(file)

    print(f"config: {config}")

    ProphetModeler.model(spark_session, config)

    print("closing spark session")
    spark_session.stop()

