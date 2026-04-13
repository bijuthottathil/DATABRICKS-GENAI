import mlflow
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
    return "\n\n".join(doc.page_content for doc in docs)


def _run_qa(inputs):
    question = inputs["query"]
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    return (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )


chain = RunnableLambda(_run_qa)
mlflow.models.set_model(chain)
