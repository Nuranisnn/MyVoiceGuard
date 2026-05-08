# Image untuk Render (WAJIB untuk upload MP3): ffmpeg dipasang di sini.
# Jangan guna "Native Python" sahaja — imej itu biasanya tiada ffmpeg → /predict-file gagal.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -n 1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
# Render set PORT masa runtime; gunicorn ikut PORT
EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 300 --workers 1 --threads 1"]
