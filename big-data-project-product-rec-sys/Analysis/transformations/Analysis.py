import dlt
from pyspark.sql.functions import col, avg, count, when, round, log1p

# ==============================================================
# STEP 1: Load Silver Data
# ==============================================================

@dlt.view(
    comment="Cleaned Silver data for analytical processing"
)
def silver_products():
    return spark.read.table("big_data_project.silver.products_cleaned")

# ==============================================================
# STEP 2: Category-level Analytics
# ==============================================================

@dlt.table(
    comment="Gold analytical layer: Category-level product statistics"
)
def gold_category_analysis():
    df = dlt.read("silver_products")

    result = (
        df.groupBy("category_name")
          .agg(
              round(avg("stars"), 2).alias("avg_rating"),
              count("*").alias("total_products"),
              round(avg("price"), 2).alias("avg_price")
          )
          .withColumn(
              "price_band",
              when(col("avg_price") < 20, "Low")
              .when((col("avg_price") >= 20) & (col("avg_price") < 100), "Medium")
              .otherwise("High")
          )
    )
    return result

# ==============================================================
# STEP 3: Product-level Feature Enrichment
# ==============================================================

@dlt.table(
    comment="Gold product-level data enriched with popularity and price insights"
)
def gold_products_with_features():
    df = dlt.read("silver_products")

    result = (
        df.withColumn("log_reviews", log1p(col("reviews")))
          .withColumn("popularity_score", round(col("stars") * col("log_reviews"), 2))
          .withColumn(
              "price_band",
              when(col("price") < 20, "Low")
              .when((col("price") >= 20) & (col("price") < 100), "Medium")
              .otherwise("High")
          )
          .select(
              "asin",
              "title",
              "category_id",
              "category_name",
              "price",
              "stars",
              "reviews",
              "popularity_score",
              "price_band",
              "imgUrl"
          )
    )
    return result
