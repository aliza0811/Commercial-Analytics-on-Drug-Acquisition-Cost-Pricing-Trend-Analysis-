# Commercial-Analytics-on-Drug-Acquisition-Cost-Pricing-Trend-Analysis-
End-to-end cloud data pipeline analyzing CMS NADAC drug pricing data (~1M records). Cleaned and transformed with PySpark on Databricks, modeled as a star schema in Snowflake, visualized in Power BI. Includes brand-vs-generic markup and price volatility metrics.

## Tech stack

| Layer | Tool |
|---|---|
| Transformation | Databricks (PySpark, serverless compute) |
| Warehouse | Snowflake (star schema) |
| BI / visualization | Power BI (DirectQuery, live connection) |
| Source data | CMS NADAC public dataset |

## CODE (DATABRICKS)

%python
path = "/Volumes/workspace/default/pharmaceutical-data"
%python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

RAW_PATH = "/Volumes/workspace/default/pharmaceutical-data/nadac-national-average-drug-acquisition-cost-08-19-2026.csv"
OUT_PATH = "/Volumes/workspace/default/pharmaceutical-data/processed/"

### 1. Read + rename + cast
df = (
    spark.read.option("header", True).option("inferSchema", False).csv(RAW_PATH)
    .withColumnRenamed("NDC Description", "ndc_description")
    .withColumnRenamed("NDC", "ndc")
    .withColumnRenamed("NADAC Per Unit", "nadac_per_unit")
    .withColumnRenamed("Effective Date", "effective_date")
    .withColumnRenamed("Pricing Unit", "pricing_unit")
    .withColumnRenamed("Pharmacy Type Indicator", "pharmacy_type_indicator")
    .withColumnRenamed("OTC", "otc_flag")
    .withColumnRenamed("Explanation Code", "explanation_code")
    .withColumnRenamed("Classification for Rate Setting", "classification")
    .withColumnRenamed("Corresponding Generic Drug NADAC Per Unit", "generic_nadac_per_unit")
    .withColumnRenamed("Corresponding Generic Drug Effective Date", "generic_effective_date")
    .withColumnRenamed("As of Date", "as_of_date")
)

df = (
    df.withColumn("nadac_per_unit", F.col("nadac_per_unit").cast("double"))
    .withColumn("generic_nadac_per_unit", F.col("generic_nadac_per_unit").cast("double"))
    .withColumn("effective_date", F.to_date("effective_date", "MM/dd/yyyy"))
    .withColumn("generic_effective_date", F.to_date("generic_effective_date", "MM/dd/yyyy"))
    .withColumn("as_of_date", F.to_date("as_of_date", "MM/dd/yyyy"))
    .withColumn("ndc", F.col("ndc").cast("string"))
    .withColumn("is_otc", F.col("otc_flag") == F.lit("Y"))
    .withColumn("is_brand", F.col("classification").isin("B", "B-ANDA", "B-BIO"))
)

### 2. Dedupe on (ndc, effective_date) — keep latest as_of_date
w = Window.partitionBy("ndc", "effective_date").orderBy(F.col("as_of_date").desc())
df = df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

### 3. Derived metrics
df = df.withColumn(
    "brand_generic_markup_pct",
    F.when(
        (F.col("is_brand")) & (F.col("generic_nadac_per_unit").isNotNull()) & (F.col("generic_nadac_per_unit") > 0),
        F.round((F.col("nadac_per_unit") - F.col("generic_nadac_per_unit")) / F.col("generic_nadac_per_unit") * 100, 2),
    ).otherwise(None),
)

w_ndc = Window.partitionBy("ndc").orderBy("effective_date")
df = df.withColumn("prior_price", F.lag("nadac_per_unit").over(w_ndc)).withColumn(
    "wow_price_change_pct",
    F.when(
        F.col("prior_price").isNotNull() & (F.col("prior_price") > 0),
        F.round((F.col("nadac_per_unit") - F.col("prior_price")) / F.col("prior_price") * 100, 2),
    ).otherwise(None),
)

### 4. Write partitioned Parquet
df.write.mode("overwrite").partitionBy("classification").parquet(OUT_PATH)

print(f"Wrote {df.count():,} rows to {OUT_PATH}")
display(df.limit(20))

%python
df.coalesce(1).write.mode("overwrite").option("header", True).csv("/Volumes/workspace/default/pharmaceutical-data/processed_csv/") 

## CODE (SNOWFLAKE)
USE DATABASE nadac_db;
USE SCHEMA nadac_db.marts;
CREATE OR REPLACE TABLE dim_drug AS
SELECT DISTINCT
    ndc,
    ndc_description,
    classification,
    is_brand,
    is_otc,
    pricing_unit
FROM nadac_pricing
QUALIFY ROW_NUMBER() OVER (PARTITION BY ndc ORDER BY effective_date DESC) = 1;
CREATE OR REPLACE TABLE dim_date AS
SELECT DISTINCT
    effective_date AS date_key,
    YEAR(effective_date) AS year,
    QUARTER(effective_date) AS quarter,
    MONTH(effective_date) AS month,
    MONTHNAME(effective_date) AS month_name,
    WEEKOFYEAR(effective_date) AS week_of_year,
    DAYNAME(effective_date) AS day_of_week
FROM nadac_pricing
WHERE effective_date IS NOT NULL;
CREATE OR REPLACE TABLE fact_nadac_pricing AS
SELECT
    ndc,
    effective_date,
    nadac_per_unit,
    generic_nadac_per_unit,
    brand_generic_markup_pct,
    wow_price_change_pct,
    explanation_code,
    as_of_date
FROM nadac_pricing
WHERE effective_date IS NOT NULL;
SELECT
    (SELECT COUNT(*) FROM fact_nadac_pricing) AS fact_rows,
    (SELECT COUNT(*) FROM dim_drug) AS drug_rows,
    (SELECT COUNT(*) FROM dim_date) AS date_rows;
