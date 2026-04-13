# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Ingest Documents with Auto Loader
# MAGIC Continuously watches for new files dropped into cloud storage and appends them to a Delta table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Database and Storage

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS rag_demo;
# MAGIC CREATE SCHEMA IF NOT EXISTS rag_demo.docs;
# MAGIC CREATE VOLUME IF NOT EXISTS rag_demo.docs.raw_files;

# COMMAND ----------

# Configuration — update to match your environment
RAW_DOCS_PATH    = "/Volumes/managedcatalog/default/data/genai/ragdemo//raw_files/"   # drop new documents here
SCHEMA_LOCATION  = "/Volumes/managedcatalog/default/data/genai/ragdemo//schema/"
CHECKPOINT_PATH  = "/Volumes/managedcatalog/default/data/genai/ragdemo//checkpoint/"
TARGET_TABLE     = "rag_demo.docs.documents"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Helper
# MAGIC Splits long documents into overlapping chunks so the retriever returns focused context.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType
import re

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"chunk_text": chunk, "chunk_index": str(len(chunks))})
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

chunk_schema = ArrayType(
    StructType([
        StructField("chunk_text",  StringType(), False),
        StructField("chunk_index", StringType(), False),
    ])
)

chunk_udf = F.udf(chunk_text, chunk_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader Streaming Ingestion

# COMMAND ----------

from pyspark.sql import functions as F

raw_stream = (
    spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")   # reads each file as one row — avoids line-by-line fragmentation
        .option("cloudFiles.schemaLocation", SCHEMA_LOCATION)
        .load(RAW_DOCS_PATH)
        .select(
            F.col("path").alias("source"),
            F.col("content").cast("string").alias("content"),
            F.current_timestamp().alias("ingested_at"),
        )
)

# Explode into chunks — each row = one retrievable chunk
chunked_stream = (
    raw_stream
        .withColumn("chunks", chunk_udf(F.col("content")))
        .withColumn("chunk", F.explode("chunks"))
        .select(
            F.concat(F.col("source"), F.lit("::"), F.col("chunk.chunk_index")).alias("chunk_id"),
            F.col("source"),
            F.col("chunk.chunk_text").alias("content"),
            F.col("chunk.chunk_index").alias("chunk_index"),
            F.col("ingested_at"),
        )
)

# Write to Delta — new files are automatically picked up forever
(
    chunked_stream
        .writeStream
        .format("delta")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .option("mergeSchema", "true")
        .outputMode("append")
        .trigger(availableNow=True)          # change to trigger(processingTime="5 minutes") for continuous
        .toTable(TARGET_TABLE)
        .awaitTermination()
)

print(f"Ingestion complete. Table: {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify ingested data
# MAGIC SELECT source, chunk_index, LEFT(content, 120) AS preview, ingested_at
# MAGIC FROM rag_demo.docs.documents
# MAGIC ORDER BY ingested_at DESC
# MAGIC LIMIT 20;