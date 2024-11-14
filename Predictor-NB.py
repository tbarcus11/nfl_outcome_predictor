# Databricks notebook source
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, BooleanType, ArrayType, DoubleType
from pyspark.sql import Row
from pyspark.sql.functions import to_date, col, when, lit, array, udf, abs
from pyspark.ml.feature import StringIndexer, MinMaxScaler
from pyspark.ml.linalg import Vectors, DenseVector

# COMMAND ----------

nfl_schema = StructType([
 StructField('schedule_date', DateType(), True),
 StructField('schedule_season', IntegerType(), True),
 StructField('schedule_week', StringType(), True),
 StructField('schedule_playoff', BooleanType(), True),
 StructField('team_home', StringType(), True),
 StructField('score_home', IntegerType(), True),
 StructField('score_away', IntegerType(), True),
 StructField('team_away', StringType(), True),
 StructField('team_favorite_id', StringType(), True),
 StructField('spread_favorite', IntegerType(), True),
 StructField('over_under_line', IntegerType(), True),
 StructField('stadium', StringType(), True),
 StructField('stadium_neutral', BooleanType(), True),
 StructField('weather_temperature', IntegerType(), True),
 StructField('weather_wind_mph', IntegerType(), True),
 StructField('weather_humidity', IntegerType(), True),
 StructField('weather_detail', StringType(), True)
])

nfl_df = spark.read.format("csv").schema(nfl_schema).option("header", True).load("/FileStore/tables/spreadspoke_scores.csv")

# COMMAND ----------

nfl_df.dtypes


# COMMAND ----------

# MAGIC %md
# MAGIC #1. Basic Label Creation:
# MAGIC Win/Loss Label: Create a column team_home_result where 1 represents a win for the home team (score_home > score_away) and 0 represents a loss (score_home < score_away). Similarly, create a team_away_result column for the away team.

# COMMAND ----------

nfl_df_win_loss = nfl_df.withColumn("team_home_result", when(col("score_home") > col("score_away"), 1).otherwise(0)).withColumn("team_away_result", when(col("score_away") > col("score_home"), 1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Team Encoding:
# MAGIC Team Code: Assign a numerical code to each team using a dictionary or by using PySpark functions. This converts team_home and team_away to numerical columns.
# MAGIC Team Strength: Create a feature to represent a team’s historical strength, like average points scored in the past X games or win rate in recent seasons.

# COMMAND ----------

all_nfl_teams = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "OAK": "Oakland Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "PICK": "Pick 'em",
    "UNK": "Unknown"
}

mapping_rows = [Row(code=k, full_name=v) for k, v in all_nfl_teams.items()]
team_mapping_df = spark.createDataFrame(mapping_rows)


# COMMAND ----------

nfl_df_win_loss = nfl_df_win_loss.fillna({
    "team_favorite_id": "UNK"
})

# COMMAND ----------

nfl_df_normalized = nfl_df_win_loss.join(
    team_mapping_df,
    nfl_df_win_loss['team_favorite_id'] == team_mapping_df['code'],
    how='left'
).drop('team_favorite_id').withColumnRenamed('full_name', 'team_favorite_id')

# COMMAND ----------

valid_team_names = list(all_nfl_teams.values())

nfl_df_normalized = nfl_df_normalized.withColumn(
    "team_favorite_id",
    when(
        col("team_favorite_id").isin(valid_team_names), col("team_favorite_id")
    ).otherwise("Unknown") 
)

# COMMAND ----------

team_index_mapping = {
    "Arizona Cardinals": 0,
    "Atlanta Falcons": 1,
    "Baltimore Ravens": 2,
    "Buffalo Bills": 3,
    "Carolina Panthers": 4,
    "Chicago Bears": 5,
    "Cincinnati Bengals": 6,
    "Cleveland Browns": 7,
    "Dallas Cowboys": 8,
    "Denver Broncos": 9,
    "Detroit Lions": 10,
    "Green Bay Packers": 11,
    "Houston Texans": 12,
    "Indianapolis Colts": 13,
    "Jacksonville Jaguars": 14,
    "Kansas City Chiefs": 15,
    "Los Angeles Chargers": 16,
    "Los Angeles Rams": 17,
    "Las Vegas Raiders": 18,
    "Miami Dolphins": 19,
    "Minnesota Vikings": 20,
    "New England Patriots": 21,
    "New Orleans Saints": 22,
    "New York Giants": 23,
    "New York Jets": 24,
    "Philadelphia Eagles": 25,
    "Pittsburgh Steelers": 26,
    "San Francisco 49ers": 27,
    "Seattle Seahawks": 28,
    "Tampa Bay Buccaneers": 29,
    "Tennessee Titans": 30,
    "Washington Commanders": 31,
    "Pick 'em": 32 ,
    "Unkown": 33
}

# COMMAND ----------

def get_team_index(team_name):
    return team_index_mapping.get(team_name, -1)
  
get_team_index_udf = udf(get_team_index)

# COMMAND ----------

nfl_df_indexed = nfl_df_normalized.withColumn("team_home_index", get_team_index_udf("team_home"))
nfl_df_indexed = nfl_df_indexed.withColumn("team_away_index", get_team_index_udf("team_away"))
nfl_df_indexed = nfl_df_indexed.withColumn("team_favorite_index", get_team_index_udf("team_favorite_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Spread and Betting Lines:
# MAGIC Spread Feature: Use spread_favorite to create a feature indicating if the favorite won or if the team covered the spread. This can be helpful for understanding potential predictors.
# MAGIC Over/Under: Create a feature showing if the total score of the game was over or under the over_under_line.

# COMMAND ----------

nfl_df_spread_over_under = nfl_df_indexed.withColumn(
    "spread_covered",
    when(
        (col("spread_favorite").isNotNull()) & (col("team_favorite_id") == col("team_home")) & 
        ((col("score_home") - col("score_away")) > col("spread_favorite")), 1
    ).when(
        (col("spread_favorite").isNotNull()) & (col("team_favorite_id") == col("team_away")) & 
        ((col("score_away") - col("score_home")) > abs(col("spread_favorite"))), 1
    ).otherwise(0)
).withColumn(
    "total_score",
    col("score_home") + col("score_away")
).withColumn(
    "over_under_result",
    when(col("over_under_line").isNotNull() & (col("total_score") > col("over_under_line")), 1)
    .otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Categorize Weather Conditions:
# MAGIC Create a simple categorical feature that classifies weather as "Good" (2), "Moderate" (1), or "Bad" (0) based on predefined conditions.
# MAGIC

# COMMAND ----------

nfl_df_weather = nfl_df_spread_over_under.withColumn(
    "weather_condition",
    when((col("weather_temperature") >= 60) & (col("weather_temperature") <= 80) & 
           (col("weather_wind_mph") <= 10) & (col("weather_humidity") <= 60), 2)
    .when((col("weather_temperature") >= 50) & (col("weather_temperature") <= 90) & 
          (col("weather_wind_mph") <= 20) & (col("weather_humidity") <= 80), 1)
    .otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Misc Processing
# MAGIC Clean up of missing values and some extra features

# COMMAND ----------

nfl_df_neutral_site = nfl_df_weather.withColumn(
    "is_neutral_site",
    when(col("stadium_neutral") == True, 1).otherwise(0)
)

# COMMAND ----------

nfl_df_filled = nfl_df_neutral_site.fillna({
    "weather_temperature": 70,  # Default to an average temperature
    "weather_wind_mph": 5,      # Default to a light wind
    "weather_humidity": 50,     # Default to average humidity
    "weather_detail": "outdoor",
    "spread_favorite": 0,
    "over_under_line": 0
})

# COMMAND ----------

proper_select = nfl_df_filled.filter(col("schedule_season") >= 2020)

# COMMAND ----------

proper_select.display()
