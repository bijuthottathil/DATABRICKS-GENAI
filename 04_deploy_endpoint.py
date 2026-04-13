# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: Deploy to Model Serving
# MAGIC Deploys the logged RAG chain as a REST endpoint.
# MAGIC **scale_to_zero = True** means you pay nothing when the endpoint is idle.

# COMMAND ----------

# MAGIC %pip install databricks-sdk databricks-vectorsearch mlflow>=2.12
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadType,
)

mlflow.set_registry_uri("databricks-uc")

# Configuration
MODEL_NAME       = "rag_demo.docs.qa_chain"
SERVING_ENDPOINT = "rag-qa-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the Latest Registered Model Version

# COMMAND ----------

client = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version)).version
print(f"Deploying model version: {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Serving Endpoint

# COMMAND ----------

import time
w = WorkspaceClient()

served_model = ServedModelInput(
    model_name=MODEL_NAME,
    model_version=str(latest_version),
    workload_type=ServedModelInputWorkloadType.CPU,   # CPU workload
    workload_size="Small",                            # Small = lowest cost
    scale_to_zero_enabled=True,                       # no idle cost
)

endpoint_config = EndpointCoreConfigInput(served_models=[served_model])

existing = [e.name for e in w.serving_endpoints.list()]

if SERVING_ENDPOINT in existing:
    # Wait for any in-progress updates to complete
    print(f"Checking endpoint state...")
    max_wait = 300  # 5 minutes
    start = time.time()
    while time.time() - start < max_wait:
        ep = w.serving_endpoints.get(SERVING_ENDPOINT)
        if ep.state.config_update.name != "IN_PROGRESS":
            break
        print(f"Endpoint update in progress, waiting...")
        time.sleep(10)
    
    print(f"Updating existing endpoint: {SERVING_ENDPOINT}")
    w.serving_endpoints.update_config(
        name=SERVING_ENDPOINT,
        served_models=[served_model],
    )
else:
    print(f"Creating new endpoint: {SERVING_ENDPOINT}")
    w.serving_endpoints.create_and_wait(
        name=SERVING_ENDPOINT,
        config=endpoint_config,
    )

print(f"Endpoint '{SERVING_ENDPOINT}' is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Deployed Endpoint via REST

# COMMAND ----------

import requests, json

token  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
url    = f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
payload = {"dataframe_records": [{"query": "What are the latest compliance requirements?"}]}

response = requests.post(url, headers=headers, data=json.dumps(payload))
response.raise_for_status()

print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Details

# COMMAND ----------

ep = w.serving_endpoints.get(SERVING_ENDPOINT)
print(f"Name    : {ep.name}")
print(f"State   : {ep.state.ready}")
print(f"URL     : {host}/serving-endpoints/{SERVING_ENDPOINT}/invocations")
