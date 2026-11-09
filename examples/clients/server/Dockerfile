FROM python:3.12-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "agent-adapter[server]==1.1.0"
WORKDIR /app
COPY server.py .
CMD ["python", "server.py"]
