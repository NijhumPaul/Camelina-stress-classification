FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script and weights into the container
COPY AI_scoring_camelina.py ./
COPY custom-20250809-123155-loss-0.11.h5 ./

ENTRYPOINT ["python", "AI_scoring_camelina.py"]
