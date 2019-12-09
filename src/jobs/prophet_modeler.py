import logging
import pickle
import time

import pandas as pd
from fbprophet import Prophet
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BinaryType, FloatType, StructType, StructField, IntegerType, TimestampType

# Modify input schema as needed if using CSV
MODEL_INPUT_SCHEMA = StructType([
    StructField("series_id", IntegerType(), True),
    StructField("dim_id", IntegerType(), True),
    StructField("start_time", TimestampType(), True),
    StructField("quantity", IntegerType(), True)
])


# Use function currying to set the parameters
# Print statements are needed to log since Python can't interface to Spark logger easily.
def model_time_series(config):
    """
    Model time series per dimensions (series_id, dim_id)
    Be sure there is sufficient data to model each time series, otherwise model may not converge.

    :param config:
    :return:
    """

    # Pandas UDF requires the output pandas dataframe schema to be defined
    output_schema = StructType([
        StructField('series_id', IntegerType(), True),
        StructField('dim_id', IntegerType(), True),
        StructField('floor', FloatType(), True),
        StructField('cap', FloatType(), True),
        StructField('model', BinaryType(), True)
    ])

    @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
    def model_time_series_udf(pdf):
        """
        User defined function for grouped sub spark dataframes converted into pandas dataframes.
        Input/output are both a pandas.DataFrame.  This cannot be an instance method.
        Be sure the order of columns and types match the defined output schema!

        Print statements are needed to log since Python can't interface to Spark logger easily.
        :param pdf: Input pandas dataframe
        :return: Output pandas dataframe
        """
        try:
            execution_time = time.time()
            series_id = int(pdf.iloc[0]['series_id'])
            dim_id = int(pdf.iloc[0]['dim_id'])

            floor = config['model']['floor']
            pdf['floor'] = floor

            cap = pdf['y'].max() * config['model']['cap_multiplier']
            pdf['cap'] = cap

            print(f"Modeling series_id: {series_id}, dim_id: {dim_id}"
                  f" with {len(pdf.index)} modeling rows")

            model = Prophet(growth='logistic', seasonality_mode='multiplicative')
            model.fit(pdf)

            data = {'series_id': [series_id],
                    'dim_id': [dim_id],
                    'floor': [floor],
                    'cap': [cap],
                    'model': [
                        pickle.dumps(model)]}  # Save the trained model by pickling the model and add to the dataframe

            output_df = pd.DataFrame(data)
            print(f"Output df series_id: {series_id}, dim_id: {dim_id}"
                  f" trained in {time.time() - execution_time}")

            return output_df

        except RuntimeError as err:
            print(f"Runtime error {err} for series_id: {series_id}, "
                  f"dim_id: {dim_id}")
            return pd.DataFrame(
                columns=['series_id', 'dim_id', 'floor', 'cap', 'model'])

    return model_time_series_udf


class ProphetModeler:
    """
    Create models to forecast quantities using Facebook Prophet model.

    Each time series has its own time series model. The collection of trained models are stored
    in a Spark dataframe for easy export/import using Spark.
    """

    def __init__(self, config, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config

    def read_input_dataframe(self, spark: SparkSession):
        """
        Reads the modeling input data
        :param spark: spark session
        :return: dataframe with input data
        """

        input_df = spark \
            .read \
            .csv(self.config['io']['input'], schema=MODEL_INPUT_SCHEMA) \
            .select('series_id', 'dim_id', 'start_time', 'quantity') \
            .withColumnRenamed("start_time", "ds") \
            .withColumnRenamed("quantity", "y")

        return input_df

    def persist_models(self, model_df: DataFrame):
        """
        Persist the models in Spark dataframe for easy export into Parquet.
        :param model_df: Spark dataframe with trained time series models
        """
        model_df \
            .write \
            .parquet(self.config['io']['models'], mode='overwrite')

    @staticmethod
    def model(spark_session, config):
        """
        Create the trained time series models
        :param spark_session:
        :param config: Dict of config
        """
        spark_session.conf.set("spark.sql.execution.arrow.enabled",
                               "true")  # needed to convert to/from Pandas dataframe
        scorer = ProphetModeler(config)
        input_df = scorer.read_input_dataframe(spark_session)

        model_df = input_df \
            .groupby('series_id', 'dim_id') \
            .apply(model_time_series(scorer.config))

        scorer.persist_models(model_df)
