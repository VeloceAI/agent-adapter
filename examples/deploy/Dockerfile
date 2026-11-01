# The HTTP tool server image: slim base, non-root user, graceful shutdown inside the pod's grace period.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install "agent-adapter[server]" "redis>=5" \
 && useradd --create-home --uid 10001 tools
WORKDIR /srv
COPY app.py .
USER 10001
EXPOSE 8000

# One process per container: scale with replicas, not workers. Never gunicorn --preload.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30", "--no-server-header"]
