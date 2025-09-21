# ---- builder ----
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN python -m venv /opt/venv && . /opt/venv/bin/activate && \
    pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app/ app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]