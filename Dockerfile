# syntax=docker/dockerfile:1

# AlphaLens uses one image for the FastAPI application and its reusable
# pipelines. The optional ML build argument is reserved for sentiment workers;
# the web API reads already-scored sentiment and stays much smaller without it.
FROM python:3.12-slim AS runtime

ARG INSTALL_ML=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/alphalens \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then \
         python -m pip install -r requirements-ml.txt; \
       fi

RUN groupadd --gid 10001 alphalens \
    && useradd --uid 10001 --gid 10001 --create-home alphalens

COPY --chown=alphalens:alphalens . .

# Generated indexes, reports, and ingestion files are mounted at runtime. The
# directories still exist in the image so liveness works without a bind mount.
RUN mkdir -p /app/data/faiss /app/data/evals /app/data/logs \
    && chown -R alphalens:alphalens /app

USER alphalens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
