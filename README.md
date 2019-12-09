Time Series Forecasting with FB Prophet and Apache Spark
========================================================
Mage Khim-Young
December, 2019

# Use Case
If you have a time series you would like to forecast, [Facebook's Prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
library is fantastic. It robustly handled seasonality, missing data, trends and trains
and scores quickly.

But what if you have a large number of different time series you need to forecast?
With the help of [Apache Spark](https://spark.apache.org/) for large scale analytics processing,
you can train and predict multiple time series and scale up processing
horizontally by modifying the Spark cluster.

[PySpark](https://spark.apache.org/docs/latest/api/python/index.html) is needed in order to use [Pandas user defined functions (UDFs)]
(https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#pandas-udfs-aka-vectorized-udfs)
which allow conversions between Spark dataframes to Pandas dataframes
with the help of [Apache Arrow](https://arrow.apache.org/).

If the historical time series are segmented by different dimensions, by grouping
for the dimensions, a time series model can be trained per time series.

# Example data
in `tests/fixtures/model-input`, the example data set has two non-temporal
dimensions. The series_id happens to be a partition column, and within the CSV
dim_id is another dimension. The timestamp and quantity correspond to the
time and y values for the time series. The schema for the CSV is defined in
`src/jobs/prophet_modeler.py`

# Modeling
When the Spark dataframe reads in the input data and a grouping clause is
applied on the non-temporal dimensions, that particular time series is
converted into a Pandas dataframe with the UDF defined. Using FB Prophet,
the model is trained on the historical data and the model itself is pickled
and returned in a new dataframe. The Spark dataframe collects all of the models per sets
of dimensions and persists them for scoring.

# Scoring
The models are read as a Spark dataframe. Similar to modeling, a Pandas UDF
is used to take the trained model and create forecast predictions based
on the given configuration. These are returned and collected in a Spark
dataframe with forecast predictions per set of dimensions.

# Developer Notes
This uses conda and PySpark so extra configuration is needed.

## Conda environment
The `environment.yml` file defines what is used for the Python code. To use
this in a Spark cluster, you should bootstrap the conda installation onto each
worker node or use a preconfigured machine image with the conda environment
preconfigured.

## Spark
This requires binary serialization between Spark and Python, so Spark 2.4+ is required along with pyarrow > 0.10

### PYSPARK_PYTHON
This environment variable should use the python executable in the conda env.
`export PYSPARK_PYTHON=/path/to/python executable for conda env`

### PYTHONPATH
Be sure to define env var PYTHONPATH to include py4j and pyspark:
`export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-xxx-src.zip:$PYTHONPATH`

Also include the `src` directory of the project to PYTHONPATH in order to run the unit tests.

## Unit testing
To run the tests, be sure to set PYTHONPATH and activate the conda environment and run
`make test`.

## Build
To build the archive, `make build` will create app.zip with the Python modules with the jobs
subdirectory. The Spark driver in PySpark has to be in a separate file outside the zip archive
(see `src/modeler_spark_driver.py and scorer_spark_driver.py`)

The app.zip and drivers are in `build/dist/*`

## App Config
The app functionality has been split into prophet_modeler which trains and persists the Prophet models and
prophet_scorer which makes forecasts from the models.

There needs to be a minimum number of observations in a time series in order for the modeler to work properly.

The modeler application configuration are YAML files (`config/*_modeler_app_config.yaml`) that needs to specify the following:

<pre>
io:
  input: [input data location]
  models: [model output location]
model:
  floor: [min value for the forecast values]
  cap_multiplier: [multiplier over the max prior values for logistic model for Prophet]
</pre>

The scorer application configuration are YAML files (`config/*_scorer_app_config.yaml`) that needs to specify the following:

<pre>
io:
  models: [model location]
  forecasts: [output for forecasts in CSV]
forecast:
  periods: [number of periods to forecast]
  frequency: [frequency of forecasts (use 15min for quarter hour for example)]
</pre>

## Spark Cluster Config
The spark cluster should not be configured for max resource allocation (described below). Only the
YARN virtual memory check should be disabled with the following config:
<pre>
 {
   "Classification": "yarn-site",
   "Properties":
     {
       'yarn.nodemanager.vmem-check-enabled': 'false'
     }
 }
</pre>
This needs to be disabled in case YARN thinks the container exceeds virtual memory limits and kills
the container.

## Spark Submit Config
Since most of the processing is done for the Prophet modeling, max resource
allocation and dynamic allocation that would typically be done for a
Spark application is not applicable here.  The spark app configuration needs to be manually tuned:

* spark.dynamicAllocation.enabled should be set to false since we are not using dynamic allocation
* spark.executor.instances should be set to the number of CORE nodes in cluster
* spark.executor.cores should be set to the number of vCPUs in a node - 1 for the driver
* spark.sql.shuffle.partitions should be set to number of executor cores * number of executor instances
* spark.speculation should be true to kill any task that takes too long compared to other tasks
* spark.speculation.multiplier set to 2, if task is twice as slow as median consider speculation
* spark.speculation.quantile set to .90, the fraction of tasks must be completed to consider speculation

About 70 to 80% of the nodes memory can be used for Spark + Python.

* spark.executor.memory is the memory for the spark app in each executor
* spark.executor.memoryOverhead is the off heap memory not used by Spark (Python in this case)

Example spark submit:
<pre>
spark-submit
      --deploy-mode client
      --master yarn
      --conf spark.dynamicAllocation.enabled=false
      --conf spark.sql.shuffle.partitions=2000
      --conf spark.executor.instances=30
      --conf spark.executor.cores=15
      --conf spark.executor.memory=8g
      --conf spark.executor.memoryOverhead=14g
      --conf spark.speculation=true
      --conf spark.speculation.multiplier=2
      --conf spark.speculation.quantile=0.90
      --py-files /path/to/app.zip
      /path/to/spark_driver.py
      /path/to/app_config.yaml
</pre>
