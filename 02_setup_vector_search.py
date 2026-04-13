# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Create Vector Search Index
# MAGIC Sets up a Databricks Vector Search endpoint and a **Delta Sync Index** that automatically
# MAGIC re-embeds and re-indexes whenever new rows appear in the source Delta table.
# MAGIC No manual embedding pipeline required.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Configuration — must match Step 1
SOURCE_TABLE     = "rag_demo.docs.documents"
VS_ENDPOINT      = "rag_endpoint"
VS_INDEX         = "rag_demo.docs.documents_index"
EMBEDDING_MODEL  = "databricks-bge-large-en"   # free built-in embedding model
PRIMARY_KEY      = "chunk_id"
CONTENT_COLUMN   = "content"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint (one-time)

# COMMAND ----------

# DBTITLE 1,Cell 6
existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]

if VS_ENDPOINT not in existing_endpoints:
    print(f"Creating endpoint: {VS_ENDPOINT}")
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("Endpoint created.")
else:
    print(f"Endpoint '{VS_ENDPOINT}' already exists — skipping creation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed on the Source Delta Table
# MAGIC Required for Delta Sync indexes to detect incremental changes.

# COMMAND ----------

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Change Data Feed enabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Sync Index (auto-syncs on new data)

# COMMAND ----------

existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]

if VS_INDEX not in existing_indexes:
    print(f"Creating index: {VS_INDEX}")
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=SOURCE_TABLE,
        index_name=VS_INDEX,
        pipeline_type="TRIGGERED",          # TRIGGERED = manual/scheduled; CONTINUOUS = real-time (higher cost)
        primary_key=PRIMARY_KEY,
        embedding_source_column=CONTENT_COLUMN,
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print("Index created. Waiting for initial sync...")
    index.wait_until_ready()
    print("Index is ready.")
else:
    print(f"Index '{VS_INDEX}' already exists.")
    index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trigger a Manual Sync (call this after bulk document loads)
# MAGIC For TRIGGERED pipelines, new data is not indexed until you call sync_index().
# MAGIC Schedule this in a Databricks Workflow to run after your ingestion job.

# COMMAND ----------

index.sync()
print("Sync triggered. Check status below.")

# COMMAND ----------

# Poll sync status
import time

while True:
    status = vsc.get_index(VS_ENDPOINT, VS_INDEX).describe()
    state = status.get("status", {}).get("detailed_state", "UNKNOWN")
    print(f"Index state: {state}")
    if state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"):
        print("Index is up to date.")
        break
    if "FAILED" in state:
        raise RuntimeError(f"Index sync failed: {status}")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test: Run a Similarity Search

# COMMAND ----------

results = index.similarity_search(
    query_text="What are the latest compliance requirements?",
    columns=["chunk_id", "source", "content"],
    num_results=3,
)

for hit in results.get("result", {}).get("data_array", []):
    chunk_id, source, content, score = hit
    print(f"Score: {score:.4f} | Source: {source}")
    print(f"  {content[:200]}\n")