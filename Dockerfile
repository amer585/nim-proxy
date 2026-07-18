# Hugging Face Spaces - Docker SDK
# Listens on port 7860 (required by HF Spaces).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 7860

# Single worker so the in-memory key cool-down state is shared across requests.
# uvloop (fast event loop) + httptools (fast HTTP parser) come from uvicorn[standard];
# forcing them on here maximizes throughput / lowers latency.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
