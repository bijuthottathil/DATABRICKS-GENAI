"""
Databricks App — RAG-based Document Q&A
Connects to the rag-qa-endpoint Model Serving endpoint and presents
a Gradio chat interface for user prompts.
"""

import os
import json
import requests
import gradio as gr
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Patch gradio_client to handle boolean JSON schemas (additionalProperties: true/false)
# Fixed upstream but not yet released in the pinned gradio version.
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gcu
    _orig = _gcu._json_schema_to_python_type

    def _patched(schema, defs=None):
        if isinstance(schema, bool):
            return "any"
        return _orig(schema, defs)

    _gcu._json_schema_to_python_type = _patched
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVING_ENDPOINT = "rag-qa-endpoint"

EXAMPLE_QUESTIONS = [
    "What are the latest compliance requirements?",
    "Summarise the key points from the recent policy update.",
    "What are the data governance policies for AI models?",
    "What are the remote working security requirements?",
    "What is the acceptable use policy for generative AI tools?",
    "What must I do if I suspect a security incident?",
]

# ---------------------------------------------------------------------------
# Endpoint call
# ---------------------------------------------------------------------------
def _get_host_and_token() -> tuple[str, str]:
    """
    Resolve Databricks host and token.
    Priority:
      1. DATABRICKS_HOST / DATABRICKS_TOKEN env vars (set automatically by Databricks Apps)
      2. Databricks SDK credential chain (profile, Azure MSI, etc.)
    """
    host  = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        w = WorkspaceClient()
        host = w.config.host
        token = w.config.token or ""

    # Ensure host has https:// scheme
    host = host.rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"

    return host, token


def call_endpoint(question: str) -> str:
    """Send a question to the serving endpoint and return the answer string."""
    host, token = _get_host_and_token()
    url = f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"dataframe_records": [{"query": question}]}

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    response.raise_for_status()

    try:
        predictions = response.json().get("predictions", [])
    except json.JSONDecodeError:
        raise ValueError(f"Endpoint returned non-JSON response (status {response.status_code}): {response.text[:200]}")

    return predictions[0] if predictions else "No answer returned from the endpoint."


# ---------------------------------------------------------------------------
# Gradio chat function
# ---------------------------------------------------------------------------
def chat(user_message: str, history: list) -> tuple[list, str]:
    """
    Called by Gradio on every user submission.
    Returns updated history and clears the input box.
    History format (Gradio 5): [{"role": "user"|"assistant", "content": str}, ...]
    """
    if not user_message.strip():
        return history, ""

    try:
        answer = call_endpoint(user_message.strip())
    except requests.HTTPError as e:
        answer = f"HTTP error calling endpoint: {e.response.status_code} — {e.response.text}"
    except Exception as e:
        answer = f"Unexpected error: {e}"

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def use_example(example: str, history: list) -> tuple[list, str]:
    """Load an example question directly into the chat."""
    return chat(example, history)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Document Q&A — Acme Financial Services",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
        #header { text-align: center; padding: 20px 0 10px 0; }
        #header h1 { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; margin: 0; }
        #header p  { color: #555; margin-top: 6px; font-size: 0.95rem; }
        #chatbox   { min-height: 420px; }
        .example-btn { font-size: 0.82rem !important; }
        #footer    { text-align: center; color: #999; font-size: 0.78rem; margin-top: 12px; }
    """,
) as demo:

    # --- Header ---
    gr.HTML("""
        <div id="header">
            <h1>Acme Financial Services — Document Q&A</h1>
            <p>Ask questions about company policies, compliance requirements, and governance frameworks.<br>
               Answers are grounded exclusively in official internal documents.</p>
        </div>
    """)

    # --- Chat area ---
    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
        type="messages",
    )

    # --- Input row ---
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask a question about company policies or compliance...",
            label="",
            scale=9,
            container=False,
            autofocus=True,
        )
        send_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)

    # --- Example questions ---
    gr.Markdown("**Try an example:**")
    with gr.Row(equal_height=True):
        for example in EXAMPLE_QUESTIONS[:3]:
            gr.Button(example, elem_classes="example-btn").click(
                fn=use_example,
                inputs=[gr.State(example), chatbot],
                outputs=[chatbot, msg_box],
            )
    with gr.Row(equal_height=True):
        for example in EXAMPLE_QUESTIONS[3:]:
            gr.Button(example, elem_classes="example-btn").click(
                fn=use_example,
                inputs=[gr.State(example), chatbot],
                outputs=[chatbot, msg_box],
            )

    # --- Clear button ---
    with gr.Row():
        clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

    gr.HTML('<div id="footer">Powered by Databricks Model Serving · Meta Llama 3.3 70B · Databricks Vector Search</div>')

    # --- Wire up events ---
    send_btn.click(fn=chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    msg_box.submit(fn=chat, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )