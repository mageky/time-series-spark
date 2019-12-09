import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, pandas_udf, PandasUDFType, udf
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType


# UDFs need to be standalone functions

# Use function currying to set the parameters
# Logging within Python doesn't work on Spark executors, so have to use print statements which
# will show up in stderr on individual workers

def forecast_time_series(config):
    """
    Forecast using trained time series model (series_id, dim_id)

    :param config:
    :return:
    """

    # Pandas UDF requires the output pandas dataframe schema to be defined
    output_schema = StructType([
        StructField('series_id', IntegerType(), True),
        StructField('dim_id', IntegerType(), True),
        StructField('ds', TimestampType(), True),
        StructField('yhat', IntegerType(), True)
    ])

    @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
    def forecast_time_series_udf(pdf):
        """
        User defined function for grouped sub spark dataframes converted into pandas dataframes.
        Input/output are both a pandas.DataFrame.  This cannot be an instance method.
        Be sure the order of columns and types match the defined output schema!
        :param pdf: Input pandas dataframe
        :return: Output pandas dataframe
        """
        try:
            series_id = int(pdf.iloc[0]['series_id'])
            dim_id = int(pdf.iloc[0]['dim_id'])
            floor = float(pdf.iloc[0]['floor'])
            cap = float(pdf.iloc[0]['cap'])
            model = pickle.loads(pdf.iloc[0]['model'])

            # If model is missing return empty dataframe for the forecast
            if model is None:
                print(f"For series_id: {series_id}, "
                      f"dim_id: {dim_id},"
                      f" no model found")
                return pd.DataFrame(columns=['series_id', 'dim_id', 'ds', 'yhat'])

            frequency = config['forecast']['frequency']

            # We want the weeks to be offset on the appropriate weekday,
            # using 'W' for pandas date_range will set it to Sundays
            if frequency == 'W':
                frequency = pd.offsets.Week()

            future_df = model.make_future_dataframe(periods=config['forecast']['periods'],
                                                    freq=frequency,
                                                    include_history=False)
            future_df['floor'] = floor
            future_df['cap'] = cap

            forecast_df = model.predict(future_df)

            # Be sure to cast yhat to integer since yhat is a float
            forecast_df = forecast_df.astype({"yhat": int})

            # Log negative values if any
            negatives = np.where(forecast_df["yhat"] < floor)
            if len(negatives[0]) > 0:
                print(f"Negative forecast values found for series_id: {series_id}, "
                      f"dim_id: {dim_id}")

                # Remove forecast values less than floor
                forecast_df["yhat"] = np.where(forecast_df["yhat"] < floor,
                                               floor,
                                               forecast_df["yhat"])

            output_df = forecast_df[['ds', 'yhat']]
            output_df['series_id'] = series_id
            output_df['dim_id'] = dim_id

            output_df = output_df[['series_id', 'dim_id', 'ds', 'yhat']]
            print(f"series_id: {series_id}; demo_id: {dim_id}; "
                  f"floor: {floor}; cap: {cap}; "
                  f"future min: {future_df['ds'].min()}; future max: {future_df['ds'].max()}; "
                  f"forecast min: {forecast_df['ds'].min()}; forecast max: {forecast_df['ds'].max()}; "
                  f"output min: {output_df['ds'].min()}; output max: {output_df['ds'].max()};")

            return output_df

        except RuntimeError as err:
            print(f"Runtime error {err} for series_id: {series_id}, "
                  f"dim_id: {dim_id}")
            return pd.DataFrame(columns=['series_id', 'dim_id', 'ds', 'yhat'])

    return forecast_time_series_udf


def extract_date(datetimestamp: datetime):
    return datetimestamp.date().strftime("%Y-%m-%d")


extract_date_udf = udf(extract_date)


class ProphetScorer:
    """
    Forecast quantities using trained Facebook Prophet model.
    """

    def __init__(self, config, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config

    def read_model_dataframe(self, spark: SparkSession):
        model_df = spark \
            .read \
            .parquet(self.config['io']['models'])

        return model_df

    @staticmethod
    def convert_forecasts(forecast_df: DataFrame):
        created_timestamp = datetime.now(timezone.utc) \
            .replace(microsecond=0) \
            .isoformat()

        return forecast_df \
            .withColumn("created_timestamp", lit(created_timestamp)) \
            .select("created_timestamp",
                    "series_id",
                    "dim_id",
                    extract_date_udf("ds").alias("forecast_date"),  # include a date without time
                    "ds",
                    "yhat") \
            .withColumnRenamed("ds", "forecast_timestamp") \
            .withColumnRenamed("yhat", "forecast_quantity")

    def write_forecasts(self, output_df: DataFrame):
        output_df \
            .write \
            .csv(self.config['io']['forecasts'], mode='overwrite', header=True)

    @staticmethod
    def score(spark_session, config):
        spark_session.conf.set("spark.sql.execution.arrow.enabled",
                               "true")  # needed to convert to/from Pandas dataframe
        scorer = ProphetScorer(config)
        model_df = scorer.read_model_dataframe(spark_session)

        forecast_df = model_df \
            .groupby('series_id', 'dim_id') \
            .apply(forecast_time_series(scorer.config))

        converted_df = scorer.convert_forecasts(forecast_df)

        scorer.write_forecasts(converted_df)
