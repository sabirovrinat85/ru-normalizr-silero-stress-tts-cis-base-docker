# ──────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-create the cache dir so COPY never fails ──
RUN mkdir -p /root/.cache

# ── Download the model ──
RUN python -c "from silero import silero_tts; silero_tts(language='ru', speaker='v5_cis_base')"

# ── Diagnostic: show where the model actually landed ──
RUN echo "=== /root/.cache tree ===" && find /root/.cache -type f | head -40 \
    && echo "=== /root/.cache/huggingface tree ===" && find /root/.cache/huggingface -type f 2>/dev/null | head -40 \
    && echo "=== /tmp (in case it cached there) ===" && find /tmp -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" 2>/dev/null | head -20
# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python env from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Model cache (broad copy to catch any layout) ──
COPY --from=builder /root/.cache /root/.cache

# Application code
COPY main.py .

ENV TTS_NUM_THREADS=4
ENV TTS_VOICES=ru_albina,ru_marat,ru_ekaterina,ru_gamat,ru_aigul,ru_alfia,ru_alfia2,ru_vika,ru_bogdan,ru_saida,ru_igor,ru_roman,tat_albina,tat_marat,chv_ekaterina,aze_gamat,bak_aigul,bak_alfia,bak_alfia2,kat_vika,udm_bogdan,uzb_saida,ukr_igor,ukr_roman
ENV TTS_DEFAULT_VOICE=ru_albina
ENV DICT_PATH=/dictionaries

VOLUME /dictionaries

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
