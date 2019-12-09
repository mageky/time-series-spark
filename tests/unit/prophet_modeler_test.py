import logging
import pytest
from pyspark.sql import SparkSession

from jobs.prophet_modeler import ProphetModeler, model_time_series


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
    spark = SparkSession.builder.master("local[*]").appName("TestModeler").getOrCreate()
    request.addfinalizer(lambda: spark.stop())

    suppress_py4j_logging()
    return spark


@pytest.fixture()
def config():
    config = {
        'io': {
            'input': 'tests/fixtures/model-input',
            'models': 'build/models',
        },
        'model': {
            'floor': 0,
            'cap_multiplier': 1.1
        }}
    return config


@pytest.fixture()
def setup(config):
    global modeler
    modeler = ProphetModeler(config)


@pytest.fixture()
def spark_input_df(spark_session):
    return modeler.read_input_dataframe(spark_session)


def test_read_dataframe(setup, spark_input_df):
    assert (spark_input_df.columns == ['series_id', 'dim_id', 'ds', 'y'])
    assert (spark_input_df.select("series_id").distinct().count() == 1)
    assert (spark_input_df.select("dim_id").distinct().count() == 2)
    assert (spark_input_df.count() == 816)


@pytest.mark.dependency()
def test_model_time_series(setup, spark_input_df):
    output_df = spark_input_df \
        .groupby('series_id', 'dim_id') \
        .apply(model_time_series(modeler.config))

    assert (output_df.count() == 2)
    assert (output_df.columns == ['series_id', 'dim_id', 'floor', 'cap', 'model'])
    assert (output_df.filter('series_id = 751 and dim_id = 91').count() == 1)
    assert (output_df.filter('series_id = 751 and dim_id = 155').count() == 1)

    modeler.persist_models(output_df)

    model_df = spark.read.parquet('./build/models')

    assert (model_df.count() == 2)
    assert (model_df.columns == ['series_id', 'dim_id', 'floor', 'cap', 'model'])
