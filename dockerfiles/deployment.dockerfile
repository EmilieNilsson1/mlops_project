FROM python:3.11-slim

ENV PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

COPY src src/

EXPOSE $PORT

CMD exec uvicorn api:app --port $PORT --host 0.0.0.0 --workers 1