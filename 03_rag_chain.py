# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Build the RAG Chain
# MAGIC Combines Databricks Vector Search retrieval with a hosted LLM using LangChain.
# MAGIC The chain is logged to MLflow for reproducibility and deployment.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch langchain langchain-community "mlflow[databricks]>=2.12"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Configuration — must match Steps 1 & 2
VS_ENDPOINT     = "rag_endpoint"
VS_INDEX        = "rag_demo.docs.documents_index"
LLM_ENDPOINT    = "databricks-meta-llama-3-3-70b-instruct"    # Databricks Foundation Model
                                                  # alternatives: "databricks-meta-llama-3-1-8b-instruct"  (cheapest)
                                                  #               "databricks-meta-llama-3.1-405b-instruct" (most powerful)
NUM_RESULTS     = 4       # chunks retrieved per query
MAX_TOKENS      = 512
TEMPERATURE     = 0.1
MODEL_NAME      = "rag_demo.docs.qa_chain"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

retriever = DatabricksVectorSearch(
    index=vs_index,
    embedding=embedding_model,
    text_column="content",
    columns=["chunk_id", "source", "content"],
).as_retriever(
    search_kwargs={"k": NUM_RESULTS}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Template

# COMMAND ----------

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, respond with "I don't have enough information to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG Chain

# COMMAND ----------

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def _run_qa(inputs):
    question = inputs["query"]
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"result": answer, "source_documents": docs, "query": question}

qa_chain = RunnableLambda(_run_qa)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Chain

# COMMAND ----------

def ask(question: str) -> None:
    result = qa_chain.invoke({"query": question})
    print(f"Q: {question}\n")
    print(f"A: {result['result']}\n")
    print("Sources:")
    seen = set()
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            print(f"  - {src}")
            seen.add(src)
    print("-" * 60)

ask("What are the latest compliance requirements?")
ask("Summarize the key points from the recent policy update.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Chain to MLflow

# COMMAND ----------

import mlflow
import os
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.langchain.autolog(disable=True)   # disable auto-logging; we log manually

# --- Models-from-code: write self-contained chain definition ----------------
_chain_code = '''import mlflow
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

VS_ENDPOINT = "rag_endpoint"
VS_INDEX = "rag_demo.docs.documents_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
NUM_RESULTS = 4
MAX_TOKENS = 512
TEMPERATURE = 0.1

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
vsc = VectorSearchClient(disable_notice=True)
vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

retriever = DatabricksVectorSearch(
    index=vs_index,
    embedding=embedding_model,
    text_column="content",
    columns=["chunk_id", "source", "content"],
).as_retriever(search_kwargs={"k": NUM_RESULTS})

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, respond with "I don't have enough information to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT,
)


def _format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)


def _run_qa(inputs):
    question = inputs["query"]
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    return (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )


chain = RunnableLambda(_run_qa)
mlflow.models.set_model(chain)
'''

chain_path = os.path.join(os.getcwd(), "chain.py")
with open(chain_path, "w") as f:
    f.write(_chain_code)

# --- Log the model ----------------------------------------------------------
sample_input  = {"query": "What are the latest requirements?"}
sample_output = qa_chain.invoke(sample_input)

signature = infer_signature(
    model_input={"query": "sample question"},
    model_output=sample_output["result"],
)

with mlflow.start_run(run_name="rag_chain_v1"):
    model_info = mlflow.langchain.log_model(
        lc_model=chain_path,
        name="rag_chain",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=sample_input,
        pip_requirements=[
            "databricks-vectorsearch",
            "langchain",
            "langchain-community",
        ],
    )

print(f"Model registered: {MODEL_NAME}")
print(f"Model URI: {model_info.model_uri}")