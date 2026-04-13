# LLM-Based Question Answering (RAG on Databricks)
<img width="1064" height="574" alt="image" src="https://github.com/user-attachments/assets/890615bc-15ac-415f-a52c-b602c9e87cbc" />


## What is this?

This project builds a system that lets you **ask questions in plain English and get accurate answers drawn directly from your own company documents** — not from the AI's general knowledge, but from the actual files you upload.

The technique behind this is called **RAG (Retrieval-Augmented Generation)**. Think of it as giving the AI a searchable library of your documents, and then asking it to answer questions by first finding the relevant pages and then summarising what it finds.

---

## Real-World Scenario: Acme Financial Services HR & Compliance Helpdesk

Imagine you work at **Acme Financial Services**. The company has hundreds of internal documents:

- IT security policies
- Data governance guidelines
- AI model governance frameworks
- Compliance requirements for 2024
- Remote working rules
- Employee handbook sections

An employee wants to know:
> *"Can I use my personal laptop to access corporate email while working from home?"*

Without this system, they would have to manually search through multiple PDF documents hoping to find the right paragraph. With this system, they just type the question and get an answer in seconds, with a reference to the exact source document.

This is the problem this POC solves.

---

## How It Works — Step by Step

```
Your Documents                  AI Brain
(PDFs, TXTs, etc.)              (LLM: Llama 3.3 70B)
       |                               |
       v                               |
 [Step 1]                              |
 Ingest & Chunk          <------------>|
 Break documents into                  |
 bite-sized passages                   |
       |                               |
       v                               |
 [Step 2]                              |
 Vector Search Index                   |
 Each passage is turned                |
 into a "fingerprint"                  |
 (embedding) so similar                |
 content can be found fast             |
       |                               |
       v                               v
 [Step 3] RAG Chain
 User asks a question
       |
       +---> Search index for the most relevant passages
       |
       +---> Hand those passages to the LLM
       |
       +---> LLM answers using ONLY what's in the passages
       |
       v
 [Step 4] REST API Endpoint
 The whole system is deployed as a live web service
 — any application can call it with a question and get an answer
```

---

## The Four Notebooks Explained

### Notebook 1 — `01_ingest_documents.ipynb` : Load Your Documents

**What it does:** Watches a cloud storage folder. Whenever a new document is dropped in, it automatically picks it up, reads it, and breaks it into overlapping passages (called "chunks") of roughly 800 words each.

**Why overlapping chunks?** So that important information that sits near the boundary between two chunks is not lost. Each chunk overlaps with its neighbour by 100 words.

**Output:** A Delta table (`rag_demo.docs.documents`) where every row is one passage from one document, tagged with its source file.

**Analogy:** Like a librarian who reads every new book, tears out the pages, and files them by topic so they're easy to retrieve later.

---

### Notebook 2 — `02_setup_vector_search.ipynb` : Build the Search Index

**What it does:** Takes every passage stored in Step 1 and converts it into a mathematical "fingerprint" called an **embedding**. These fingerprints are stored in a Vector Search Index that can find semantically similar content in milliseconds.

**Why embeddings?** Traditional keyword search fails if the user asks *"remote work laptop rules"* but the document says *"BYOD policy for home office"*. Embeddings understand meaning, not just exact words, so they match the right content regardless of how the question is phrased.

**Output:** A live, searchable vector index (`rag_demo.docs.documents_index`) backed by Databricks' built-in `databricks-bge-large-en` embedding model. The index automatically updates whenever new documents are ingested.

**Analogy:** Like building a smart card-catalogue for the library — one that understands what you mean, not just what you say.

---

### Notebook 3 — `03_rag_chain.ipynb` : Wire the Retriever to the LLM

**What it does:** Combines the vector search retriever (Step 2) with a powerful Large Language Model (Meta Llama 3.3 70B) to build the full question-answering pipeline.

When a question arrives:
1. The retriever finds the 4 most relevant passages from the index.
2. Those passages are inserted into a prompt alongside the question.
3. The LLM reads only those passages and writes a grounded answer.
4. If the answer is not in the documents, it says so — it never makes things up.

The chain is logged to MLflow so it is versioned, reproducible, and ready to deploy.

**Output:** A registered model (`rag_demo.docs.qa_chain`) in the Unity Catalog model registry.

**Analogy:** Like a very well-read assistant who, before answering your question, quickly pulls out the three most relevant pages from the filing cabinet and reads them — and is honest when the answer is not there.

> For a full technical breakdown of how the chain is built and why it is logged to MLflow, see the **RAG Chain — Deep Dive** section below.

---

### Notebook 4 — `04_deploy_endpoint.ipynb` : Go Live

**What it does:** Deploys the registered model as a live REST API endpoint (`rag-qa-endpoint`) on Databricks Model Serving. Any application — a chatbot UI, a Slack bot, a mobile app — can now send a question via HTTP and receive an answer.

`scale_to_zero = True` means the endpoint shuts down automatically when idle, so you only pay when it is actually being used.

**Output:** A live HTTPS endpoint you can call like this:

```json
POST /serving-endpoints/rag-qa-endpoint/invocations
{
  "dataframe_records": [
    { "query": "What are the remote working security requirements?" }
  ]
}
```

```json
{
  "predictions": [
    "VPN usage is now mandatory for all remote access to corporate systems.
     Personal devices must have endpoint protection software approved by
     the IT Security team installed and active..."
  ]
}
```

---

## Sample Questions and Answers (from the POC run)

| Question | Answer (summary) |
|---|---|
| What are the latest compliance requirements? | Outlines Acme Financial Services 2024 updates: data privacy, financial regulations, cybersecurity, AI governance... |
| Summarise the key points from the recent policy update. | Remote Work Security Policy (VPN mandatory, BYOD rules), Acceptable Use Policy for GenAI tools... |
| What are the data governance policies for AI models? | Training data must be approved by Data Owner, comply with Data Governance Policy, personal data requires lawful basis... |
| What are the remote working conditions? | VPN required, BYOD devices need approved endpoint protection, unmanaged devices must use VDI... |

---

## Technology Stack

| Component | Technology |
|---|---|
| Data platform | Databricks (Azure) |
| Document storage | Delta Lake + Unity Catalog Volumes |
| Streaming ingestion | Databricks Auto Loader |
| Vector database | Databricks Vector Search |
| Embedding model | `databricks-bge-large-en` (free, built-in) |
| LLM | Meta Llama 3.3 70B Instruct (Databricks Foundation Models) |
| Orchestration | LangChain |
| Model registry & tracking | MLflow + Unity Catalog |
| Serving | Databricks Model Serving (REST endpoint) |
| User interface | Gradio (Databricks App) |

---

## LangChain — How the Q&A Chain Is Wired Together

LangChain acts as the **glue layer** between the three Databricks services (Vector Search, Embedding Model, LLM). It does not do any AI itself — it provides pre-built connectors and a standard interface to wire them into a single callable pipeline.

### What each LangChain component does

| Component | LangChain Class | Role |
|---|---|---|
| Vector Store | `DatabricksVectorSearch` | Wraps the Vector Search index so it can be queried like a standard retriever |
| Embedding Model | `DatabricksEmbeddings` | Converts the user's question into a vector before searching |
| LLM | `ChatDatabricks` | Sends the assembled prompt to Llama 3.3 70B and gets the answer |
| Prompt | `PromptTemplate` | Assembles the retrieved chunks + question into the exact text sent to the LLM |
| Pipeline wrapper | `RunnableLambda` | Packages the whole flow as a single callable object that MLflow can log and deploy |
| Output cleaner | `StrOutputParser` | Strips the LLM's response object down to a plain string |

### Chain flow diagram

```
User Question
      |
      v
┌─────────────────────────────┐
│  DatabricksEmbeddings        │  Convert question → vector fingerprint
│  (databricks-bge-large-en)  │
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│  DatabricksVectorSearch      │  Find top-4 most relevant chunks
│  (rag_demo.docs.documents    │  from the Vector Search index
│   _index)                   │
└─────────────┬───────────────┘
              |
              v  (4 document chunks returned)
┌─────────────────────────────┐
│  PromptTemplate              │  Build the prompt:
│                              │  "Answer using ONLY this context:
│                              │   {chunk1} {chunk2} {chunk3} {chunk4}
│                              │   Question: {user_question}"
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│  ChatDatabricks              │  Send prompt to Llama 3.3 70B
│  (Llama 3.3 70B Instruct)   │  Temperature: 0.1 (factual, low creativity)
│                              │  Max tokens: 512
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│  StrOutputParser             │  Extract plain text answer from LLM response
└─────────────┬───────────────┘
              |
              v
         Final Answer
   (grounded in your documents)
```

### Why `temperature=0.1`?

The LLM is deliberately set to a very low temperature (0.1 out of 1.0). Higher temperature = more creative and varied answers. Lower = more precise and factual. For a compliance and policy helpdesk, you want the AI to stick closely to what the documents actually say, not improvise.

### The "only use the context" rule

The prompt instructs the LLM: *"Answer using ONLY the provided context. If the answer cannot be found, say so."* This prevents the model from mixing in its own general knowledge (which could be wrong or outdated) with what your documents say.

---

## RAG Chain — Deep Dive

This section walks through exactly what `03_rag_chain.py` builds, step by step, and explains why each decision was made — including why the chain must be logged to MLflow before it can be deployed.

---

### Step 1 — Build the Retriever

```python
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

retriever = DatabricksVectorSearch(
    index=vs_index,
    embedding=embedding_model,
    text_column="content",
    columns=["chunk_id", "source", "content"],
).as_retriever(search_kwargs={"k": 4})
```

The retriever is a two-part object:

- **`DatabricksEmbeddings`** — when a question arrives, this converts it into a vector fingerprint using the same embedding model that was used to index the documents. Both must use the same model so the similarity comparison is meaningful.
- **`DatabricksVectorSearch`** — takes that fingerprint and searches the vector index for the 4 closest matching chunks (`k=4`). It returns the actual text of those chunks along with their `chunk_id` and `source` filename.

Why `k=4`? Enough context for the LLM to reason over without overwhelming the prompt. Too few = incomplete answers. Too many = the LLM loses focus or exceeds the token limit.

---

### Step 2 — Connect the LLM

```python
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=512,
    temperature=0.1,
)
```

- **`max_tokens=512`** — caps the length of the answer. Prevents runaway responses.
- **`temperature=0.1`** — keeps answers precise and factual. At 0 the model always gives the most likely next word. At 1 it is more creative and varied. For a compliance helpdesk, you want 0.1 — close to factual but not completely rigid.

---

### Step 3 — Define the Prompt Template

```python
RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, respond with "I don't have enough information to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
```

The prompt has two placeholders that get filled in at runtime:

- `{context}` — the 4 retrieved chunks joined together as plain text
- `{question}` — the user's original question

The instruction **"using ONLY the provided context"** is the critical safety rule. Without it, the LLM would blend your document content with its general training knowledge, which could produce plausible-sounding but incorrect answers. With it, if the answer is not in the retrieved chunks, the model says so explicitly rather than guessing.

---

### Step 4 — Assemble the Chain

```python
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def _run_qa(inputs):
    question = inputs["query"]
    docs     = retriever.invoke(question)
    context  = _format_docs(docs)
    answer   = (prompt | llm | StrOutputParser()).invoke(
                   {"context": context, "question": question}
               )
    return {"result": answer, "source_documents": docs, "query": question}

qa_chain = RunnableLambda(_run_qa)
```

`_run_qa` is the heart of the system. Every time a question arrives, it:

1. Calls the retriever → gets back 4 document objects
2. Flattens them into a single string with `_format_docs`
3. Runs `prompt | llm | StrOutputParser()` — fills the template, sends it to the LLM, strips the response to plain text
4. Returns the answer **and** the source documents — so you always know which files the answer came from

`RunnableLambda` wraps this plain Python function into a LangChain-compatible runnable so it can be logged and deployed by MLflow.

---

### Step 5 — Write `chain.py` (Models-from-Code Pattern)

```python
chain_path = os.path.join(os.getcwd(), "chain.py")
with open(chain_path, "w") as f:
    f.write(_chain_code)
```

Before logging to MLflow, the chain logic is written out to a standalone file called `chain.py`. This is MLflow's **models-from-code** pattern.

**Why not just pickle the chain object directly?**

LangChain objects contain live connections — to the vector search endpoint, to the LLM endpoint. You cannot serialise a live network connection into a pickle file. Instead, MLflow logs the *source code* (`chain.py`) and re-executes it at serve time inside the model container, which re-establishes the connections fresh in the deployment environment.

This also means the deployed model is always environment-aware — it connects to the correct Databricks endpoints automatically when it boots up, without any hardcoded credentials.

---

### Step 6 — Log the Chain to MLflow

```python
signature = infer_signature(
    model_input={"query": "sample question"},
    model_output=sample_output["result"],
)

with mlflow.start_run(run_name="rag_chain_v1"):
    model_info = mlflow.langchain.log_model(
        lc_model=chain_path,
        registered_model_name="rag_demo.docs.qa_chain",
        signature=signature,
        input_example=sample_input,
        pip_requirements=["databricks-vectorsearch", "langchain", "langchain-community"],
    )
```

#### Why log to MLflow?

This is the step that turns a notebook experiment into a deployable, versioned product. Here is what each part gives you:

| What is logged | Why it matters |
|---|---|
| **`chain.py` source file** | The exact code that runs when the endpoint receives a question — no surprises in production |
| **`signature`** | Enforces the input/output contract: input must be `{"query": string}`, output must be a string. Databricks Model Serving rejects malformed requests before they reach your code |
| **`input_example`** | A sample request stored alongside the model — used to smoke-test the endpoint after deployment and shown in the Model Serving UI |
| **`pip_requirements`** | The exact Python packages needed. MLflow builds the deployment container with these — guarantees the same library versions that were tested are what runs in production |
| **`registered_model_name`** | Saves the model to Unity Catalog (`rag_demo.docs.qa_chain`) with a version number. Notebook 4 fetches the latest version from here to deploy |
| **Run name `rag_chain_v1`** | Every time you re-run this notebook a new MLflow run is created, so you have a full history of every version — when it was built, what parameters it used, what the test output was |

#### What `infer_signature` does

`infer_signature` inspects a real input and a real output from a live test call and records the expected shapes and types. This becomes a contract: if something calls the endpoint with the wrong field name or wrong type, it is rejected immediately with a clear error rather than silently failing inside the chain.

---

### End-to-end flow inside `_run_qa`

```
User sends: {"query": "What are the remote working security requirements?"}
                              |
                              v
        retriever.invoke(question)
        → embedding model converts question to vector
        → vector search finds 4 nearest chunks
        → returns: [chunk1, chunk2, chunk3, chunk4]
                              |
                              v
        _format_docs(docs)
        → joins 4 chunks into one block of text:
          "VPN usage is now mandatory...
           Personal devices must have...
           Unmanaged devices must use VDI...
           IT Security team must approve..."
                              |
                              v
        prompt.invoke({"context": <above text>, "question": <original question>})
        → fills RAG_PROMPT template
                              |
                              v
        llm.invoke(filled_prompt)
        → Llama 3.3 70B reads the 4 chunks and the question
        → writes a grounded answer in plain English
                              |
                              v
        StrOutputParser()
        → strips response object → plain string
                              |
                              v
Returns: {
  "result": "VPN usage is mandatory for all remote access...",
  "source_documents": [chunk1, chunk2, chunk3, chunk4],
  "query": "What are the remote working security requirements?"
}
```

---

## Gradio App — User Interface

The `app.py` file is a **Databricks App** that provides a browser-based chat interface on top of the REST endpoint. Any employee can open the URL, type a question, and get an answer — no coding required.

### What the app looks like

- A chat window that keeps the full conversation history
- 6 pre-built example question buttons so users can get started instantly
- A clear button to reset the conversation
- Branded for Acme Financial Services

### How it connects to the endpoint

The app calls `rag-qa-endpoint` via HTTP POST. Authentication is handled automatically — Databricks Apps inject `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables at runtime, so no credentials need to be hardcoded or managed manually.

```
User types question
       |
       v
app.py (Gradio — Databricks App)
       |
       | POST /serving-endpoints/rag-qa-endpoint/invocations
       | Authorization: Bearer <auto-injected token>
       |
       v
rag-qa-endpoint (Databricks Model Serving)
       |
       v
Answer displayed in chat UI
```

### Files

| File | Purpose |
|---|---|
| `app.py` | Gradio chat UI — calls the endpoint and renders responses |
| `app.yaml` | Databricks App configuration — tells the platform to run `python app.py` |
| `requirements.txt` | Python dependencies including `gradio>=4.0.0` |

---

## What Each File Does — Plain English

---

### `01_ingest_documents.py` — The Document Reader

**What it is:** The starting point of the whole pipeline.

**What it does in plain English:** Imagine you have a big pile of company PDFs sitting in a cloud folder. This file acts like an automated librarian — it watches that folder, picks up every document, reads through it, and breaks it into smaller bite-sized pieces (around 800 words each, with a small overlap so nothing important falls through the cracks). Each piece gets stored in a database table, tagged with the name of the file it came from.

**Why it matters:** Without this step, the AI would have no documents to search through. Every other step depends on this one running first.

---

### `02_setup_vector_search.py` — The Smart Card Catalogue

**What it is:** Builds the search engine that understands meaning, not just keywords.

**What it does in plain English:** After the documents are broken into pieces (Step 1), this file takes each piece and converts it into a unique mathematical "fingerprint" — called an embedding. These fingerprints are stored in a special index that can find content by what it *means*, not just what it *says*. So a search for "remote work laptop rules" will still find the paragraph about "BYOD policy for home office", even though the words are completely different.

**Why it matters:** This is what makes the system smart. Without this, you'd only get results for exact word matches, which would miss most of the relevant answers.

---

### `03_rag_chain.py` — The Brain Builder

**What it is:** The file that wires the search engine to the AI and packages them together.

**What it does in plain English:** This file builds the actual question-answering logic — it connects the smart search (Step 2) to the AI language model (Llama 3.3 70B). When a question comes in, it first fetches the 4 most relevant document passages, then hands them to the AI along with the question, and the AI answers using *only* those passages. It also writes a companion file called `chain.py` (see below) and saves the whole packaged system into an MLflow model registry so it is versioned and ready to deploy.

**Why it matters:** This is where the intelligence of the system lives. It is the step that turns a search index into a full question-answering product.

---

### `chain.py` — The Packaged Brain (Auto-Generated)

**What it is:** The self-contained brain of the Q&A system, ready to be deployed.

**What it does in plain English:** This file is *not* written by hand — it is automatically created by `03_rag_chain.py` every time that script runs. Think of it as the final, clean, packaged version of the question-answering logic that gets handed to the deployment system. It holds all the connections — to the vector search index, to the embedding model, to the AI — in one tidy file. MLflow logs this file as the model artifact, so what gets deployed is always an exact snapshot of what was tested.

**Why it matters:** It ensures that exactly what you tested is what gets deployed. Nothing more, nothing less.

---

### `04_deploy_endpoint.py` — The Launch Button

**What it is:** Deploys the packaged model as a live web service.

**What it does in plain English:** Once the Q&A chain is built and saved (Step 3), this file takes it and publishes it as a live REST API — a web address that any application can call by sending a question and getting an answer back. It also sets the endpoint to switch off automatically when nobody is using it, so you only pay for what you use.

**Why it matters:** This is what turns a local experiment into a real product. After this step, any app — a chatbot, a web form, a Slack bot — can query your document library without knowing anything about the underlying AI or data.

---

### `app.py` — The User Interface

**What it is:** A browser-based chat window for non-technical users.

**What it does in plain English:** This file builds the web interface — the part employees actually see and interact with. It is a simple chat window (built with a tool called Gradio) where someone can type a question, click send, and read the answer. It also has six pre-built example questions as buttons, so users can get started immediately without having to think of what to ask. Behind the scenes, every question is silently sent to the live endpoint (Step 4) and the answer is displayed in the chat. No coding needed by the end user.

**Why it matters:** All the technical complexity is hidden away. From the user's point of view, they are just chatting with a helpful assistant that knows their company's documents inside out.

---

## Running the POC

### Step 1 — Build the pipeline (run notebooks in order)

Each notebook is self-contained and begins with a configuration block at the top where you set paths and endpoint names to match your Databricks environment.

```
01_ingest_documents.ipynb     → load and chunk documents
02_setup_vector_search.ipynb  → create embedding index
03_rag_chain.ipynb            → build and test the Q&A chain
04_deploy_endpoint.ipynb      → deploy as live REST API
```

### Step 2 — Deploy the Gradio App

1. In your Databricks workspace, go to **Apps** in the left sidebar
2. Click **Create App**
3. Point it to this folder (or upload `app.py`, `app.yaml`, and `requirements.txt`)
4. Databricks installs dependencies and starts the app automatically
5. Copy the app URL and share it with your users — done

> **Note:** Steps 2–4 of the pipeline and the App deployment require a Databricks workspace with Unity Catalog enabled and access to Databricks Foundation Models.

---

## Kubernetes Deployment (Minikube)

As an alternative to running `app.py` as a Databricks App, the Gradio UI can be deployed as a container on any Kubernetes cluster, including a local Minikube instance. This is useful for local testing, demos, or running the UI outside of Databricks.

### Files added for Kubernetes

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the container image — installs only what `app.py` needs (Gradio, FastAPI, uvicorn, requests, databricks-sdk, python-dotenv) |
| `k8s/deployment.yaml` | Kubernetes `Deployment` + `Service` manifest |
| `.env` | Local credentials file — **never committed** (already in `.gitignore`) |

### Architecture

```
Browser
   |
   | http://192.168.49.2:30800
   v
┌─────────────────────────────────────────┐
│  Minikube Cluster                        │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  Service (NodePort :30800)        │   │
│  └──────────────┬───────────────────┘   │
│                 │                        │
│  ┌──────────────▼───────────────────┐   │
│  │  Pod: gradio-app                  │   │
│  │  uvicorn + FastAPI + Gradio 5     │   │
│  │  Port 8000                        │   │
│  │                                   │   │
│  │  Env injected from K8s Secret:    │   │
│  │    DATABRICKS_HOST                │   │
│  │    DATABRICKS_TOKEN               │   │
│  │    SERVING_ENDPOINT               │   │
│  └──────────────┬───────────────────┘   │
└─────────────────│───────────────────────┘
                  │
                  │ HTTPS POST (Bearer token)
                  v
   Databricks Model Serving
   rag-qa-endpoint
```

### Prerequisites

- Docker Desktop running
- Minikube running (`minikube status`)
- `kubectl` configured to point at Minikube

### Step-by-step deployment

**1. Fill in credentials**

Edit `.env` with your real values:
```
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
SERVING_ENDPOINT=rag-qa-endpoint
```

**2. Build the Docker image inside Minikube's Docker daemon**

```bash
eval $(minikube docker-env)
docker build -t gradio-app:latest .
```

`eval $(minikube docker-env)` points your shell's Docker client at Minikube's internal Docker engine so the image is available to the cluster without pushing to a registry. `imagePullPolicy: Never` in the manifest ensures Kubernetes uses this local image.

**3. Create the Kubernetes Secret**

Credentials are stored as a Kubernetes Secret — never baked into the image or manifest.

```bash
source .env
kubectl create secret generic databricks-creds \
  --from-literal=host="$DATABRICKS_HOST" \
  --from-literal=token="$DATABRICKS_TOKEN"
```

**4. Deploy**

```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/gradio-app
```

**5. Open in browser**

```bash
minikube service gradio-app
```

Or navigate directly to `http://192.168.49.2:30800`.

### Updating after code changes

Any change to `app.py` requires rebuilding the image and restarting the pod:

```bash
eval $(minikube docker-env)
docker build -t gradio-app:latest .
kubectl rollout restart deployment/gradio-app
```

### Updating credentials

If you rotate your PAT token or change workspace:

```bash
kubectl delete secret databricks-creds
source .env   # after updating .env
kubectl create secret generic databricks-creds \
  --from-literal=host="$DATABRICKS_HOST" \
  --from-literal=token="$DATABRICKS_TOKEN"
kubectl rollout restart deployment/gradio-app
```

### Key design decisions

| Decision | Reason |
|---|---|
| `imagePullPolicy: Never` | Uses the locally built Minikube image — no Docker Hub or registry needed |
| Credentials in K8s Secret | Secrets are base64-encoded and injected as env vars at runtime — never hardcoded in the image or manifest |
| TCP socket health probes (not HTTP) | Gradio 5's schema introspection logs non-fatal TypeErrors on some HTTP routes; TCP probes check only that the port is open, avoiding false unhealthy signals |
| FastAPI + uvicorn instead of `demo.launch()` | Gradio's `launch()` does a localhost self-ping after startup which fails in Kubernetes networking — `gr.mount_gradio_app` + uvicorn bypasses this entirely |
| `timeout=300` on endpoint calls | Databricks Model Serving with `scale_to_zero=True` can take up to 3 minutes to wake from idle — a short timeout would cause false errors on first request |

### Cold start behaviour

The Databricks endpoint has `scale_to_zero=True` — it shuts down after a period of inactivity to avoid idle costs. The **first question after a period of inactivity** will take up to 3 minutes to respond while the endpoint wakes up. Subsequent questions are fast. This is expected behaviour, not a bug.

---

## Recent Changes

### Chunking fix moved to ingestion stage (`01_ingest_documents.py`)

**Problem:** The original ingestion used `cloudFiles.format = "text"`, which reads documents **line by line** rather than as whole files. This produced hundreds of tiny fragments (averaging ~113 characters each) — far too small for the LLM to reason over meaningfully. A workaround patch had been added inside `04_deploy_endpoint.py` to re-ingest documents with better settings after the fact.

**Fix:** The proper chunking settings are now applied at the source in `01_ingest_documents.py`. The workaround in `04_deploy_endpoint.py` has been removed.

| Setting | Before | After |
|---|---|---|
| File read format | `cloudFiles.format = "text"` (line-by-line) | `cloudFiles.format = "binaryFile"` (whole file per row) |
| Chunk size | 512 words | 800 words |
| Overlap | 64 words | 100 words |
| `chunk_index` type | `int` (mismatched with `StringType()` schema) | `str` (correct) |

**Files changed:**
- `01_ingest_documents.py` — format, chunk size, overlap, and type all fixed
- `04_deploy_endpoint.py` — re-ingest workaround block removed (re-ingest, overwrite, vector sync, extended test)

---

### README: File descriptions section added

A plain-English section — **"What Each File Does"** — was added to the README explaining each `.py` file in the project in non-technical terms, covering what it is, what it does, and why it matters.

---

### README: LangChain implementation section added

A new section was added explaining how LangChain wires the three Databricks services together (Vector Search, Embedding Model, LLM), including a step-by-step chain flow diagram and notes on the `temperature=0.1` setting and the "only use context" prompt rule.

