import logging
import re
from datetime import datetime
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from jobs.prophet_scorer import ProphetScorer, forecast_time_series


def suppress_py4j_logging():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    """ fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """
    global spark
    spark = SparkSession.builder.master("local[*]").appName("TestScorer").getOrCreate()
    request.addfinalizer(lambda: spark.stop())

    suppress_py4j_logging()
    return spark


@pytest.fixture()
def config():
    config = {
        'io': {
            'models': 'build/models',
            'forecasts': 'build/forecasts',
        },
        'forecast': {
            'periods': 40,
            'frequency': '15min'
        }}
    return config


@pytest.fixture()
def setup(config):
    global scorer
    scorer = ProphetScorer(config)


@pytest.fixture()
def spark_model_df(spark_session):
    return scorer.read_model_dataframe(spark_session)


@pytest.fixture()
def spark_forecast_df(spark_session):
    schema = StructType([StructField("series_id", IntegerType()),
                         StructField("dim_id", IntegerType()),
                         StructField("ds", TimestampType()),
                         StructField("yhat", IntegerType())
                         ])

    test_list = [
        (101, 66, datetime.strptime('2015-07-05 10:15:00', '%Y-%m-%d %H:%M:%S'), 873242)
    ]

    return spark_session.createDataFrame(data=test_list, schema=schema)


def test_convert_forecasts(setup, spark_forecast_df):
    output_df = scorer.convert_forecasts(spark_forecast_df)

    timestamp_regex = re.compile(r'^([0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T'
                                 r'(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\+00:00)$')
    assert (timestamp_regex.match(output_df.collect()[0][0]))
    assert (output_df.collect()[0][1] == 101)
    assert (output_df.collect()[0][2] == 66)
    assert (output_df.collect()[0][3] == '2015-07-05')
    assert (output_df.collect()[0][4] == datetime(2015, 7, 5, 10, 15))
    assert (output_df.collect()[0][5] == 873242)


def test_read_model_dataframe(setup, spark_model_df):
    assert (spark_model_df.columns == ['series_id', 'dim_id', 'floor', 'cap',
                                       'model'])
    assert (spark_model_df.select("series_id").distinct().count() == 1)
    assert (spark_model_df.select("dim_id").distinct().count() == 2)
    assert (spark_model_df.count() == 2)


@pytest.mark.dependency(depends=["test_model_time_series"])
def test_forecast_time_series(setup, spark_model_df):
    output_df = spark_model_df \
        .groupby('series_id', 'dim_id') \
        .apply(forecast_time_series(scorer.config))

    assert (output_df.count() == 80)
    assert (output_df.columns == ['series_id', 'dim_id', 'ds', 'yhat'])
    assert (output_df.filter('series_id = 751 and dim_id = 91').count() == 40)
    assert (output_df.filter('series_id = 751 and dim_id = 155').count() == 40)

    converted_df = scorer.convert_forecasts(output_df)

    scorer.write_forecasts(converted_df)

    read_output_df = spark.read.csv('./build/forecasts', header=True)

    assert (read_output_df.columns == ['created_timestamp',
                                       'series_id',
                                       'dim_id',
                                       'forecast_date',
                                       'forecast_timestamp',
                                       'forecast_quantity'])
    assert (read_output_df.count() == 80)
