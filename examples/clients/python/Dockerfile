FROM python:3.12-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "mcp>=2.3,<3"
WORKDIR /app
COPY client.py .
CMD ["python", "client.py"]
