# LinkedIn Post

---

Do you want to set up a Databricks RAG system on the Databricks Free Edition? Try it below.

On top of that, get hands-on with Databricks App Deployment — and go one step further by deploying your Databricks App on Kubernetes.

My latest post walks through building a production-ready RAG system end to end: document ingestion with Auto Loader, semantic search with Databricks Vector Search, a LangChain chain backed by Llama 3.3 70B, and a Gradio chat UI that you can ship two ways — as a native Databricks App or as a container running on Minikube/Kubernetes.

The idea is straightforward: use what Databricks gives you for free (BGE embeddings, Foundation Models, Vector Search, MLflow) and focus your energy on the parts that actually matter — chunk quality, grounded prompts, and handling cold-start latency gracefully.

The Kubernetes path is a good complement if you need to run the UI outside Databricks — same app.py, same endpoint, different runtime. The post covers the Dockerfile, the Kubernetes manifests, and the design decisions that took the most debugging to get right.

If you are building with the Databricks Free Edition and want a concrete RAG project to get started with, this is a good place to begin. Full post here: [Link to article]

Happy reading!

hashtag#dataengineering hashtag#databricks hashtag#generativeai hashtag#rag hashtag#llm hashtag#kubernetes hashtag#mlflow hashtag#langchain
