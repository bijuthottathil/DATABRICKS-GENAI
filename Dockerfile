FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "gradio==5.9.1" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.27.0" \
    requests>=2.31.0 \
    databricks-sdk>=0.28.0 \
    python-dotenv>=1.0.0

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
