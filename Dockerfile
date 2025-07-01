FROM python:3.11.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/backend/services/models

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git htop\
 && rm -rf /var/lib/apt/lists/*

# Copy và install từng nhóm packages
COPY requirements.txt .

# Multi-stage pip install để giảm layer size
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

COPY . .

CMD ["bash"]
