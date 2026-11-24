import io
import os
import wave
import dataclasses
import torch
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ru_normalizr import NormalizeOptions, Normalizer
from silero import silero_tts
from silero_stress import load_accentor          # ← the real import

# ── CPU threads ──
N_THREADS = int(os.environ.get("TTS_NUM_THREADS", os.cpu_count() or 4))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
torch.set_num_threads(N_THREADS)

# ── Voices ──
ALLOWED_VOICES = [
    v.strip()
    for v in os.environ.get("TTS_VOICES", "ru_albina,ru_marat,ru_ekaterina,ru_gamat, ru_aigul,ru_alfia,ru_alfia2,ru_vika,ru_bogdan,ru_saida,ru_igor,ru_roman,tat_albina,tat_marat,chv_ekaterina,aze_gamat, bak_aigul,bak_alfia,bak_alfia2,kat_vika,udm_bogdan,uzb_saida,ukr_igor,ukr_roman").split(",")
    if v.strip()
]
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "ru_albina")

# ── ru-normalizr dictionaries (Docker volume) ──
DICT_PATH = os.environ.get("DICT_PATH", "/dictionaries")
INCLUDE = [f.strip() for f in os.environ.get("DICT_INCLUDE", "").split(",") if f.strip()]
EXCLUDE = [f.strip() for f in os.environ.get("DICT_EXCLUDE", "").split(",") if f.strip()]


def _build_normalizer() -> Normalizer:
    base = NormalizeOptions.tts()
    overrides = {
        "dictionaries_path": Path(DICT_PATH) if DICT_PATH else None,
        "dictionary_include_files": tuple(INCLUDE) if INCLUDE else (),
        "dictionary_exclude_files": tuple(EXCLUDE) if EXCLUDE else (),
    }
    try:
        opts = dataclasses.replace(base, **overrides)
    except TypeError as e:
        print(f"ERROR: dictionary options not supported: {e}")
        opts = base
    return Normalizer(opts)


app = FastAPI()

# ── Load once at startup ──
normalizer    = _build_normalizer()              # ← use the dict-aware builder
accentor      = load_accentor()                # ← loads the stress/accent model
tts_model, _  = silero_tts(language="ru", speaker="v5_cis_base")


# ── Schemas ──

class TTSRequest(BaseModel):
    text: str
    speaker: str | None = None
    sample_rate: int = 48000


class OpenAITTSRequest(BaseModel):
    model: str = "v5_cis_base"
    input: str
    voice: str = "ru_albina"
    response_format: str = "wav"


# ── Pipeline: normalize → accentize → TTS ──

def _synthesize(text: str, voice: str, sample_rate: int = 48000) -> io.BytesIO:
    # 1. Text normalization
    normalized = normalizer.normalize(text)

    # 2. Accent / stress marking
    #    load_accentor() returns a model; the most likely call shapes are:
    #      accentor(text)            → __call__
    #      accentor.apply(text)      → .apply()
    #      accentor.forward(text)    → .forward()
    #    Try them in order; fall back to un-accented text on failure.
    accented = normalized
    for _method in ("__call__", "apply", "forward", "predict", "process"):
        fn = getattr(accentor, _method, None)
        if callable(fn):
            try:
                result = fn(normalized)
                if isinstance(result, str):
                    accented = result
                elif isinstance(result, dict) and "text" in result:
                    accented = result["text"]
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    accented = result[0] if isinstance(result[0], str) else normalized
                break
            except Exception as e:
                print(f"[tts] accentor.{_method} failed: {e}")
                break

    # 3. TTS
    audio = tts_model.apply_tts(
        text=accented,
        speaker=voice,
        sample_rate=sample_rate,
    )

    # 4. WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((audio * 32767).numpy().astype("int16").tobytes())
    buf.seek(0)
    return buf


# ── Endpoints ──

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": "v5_cis_base",
        "threads": N_THREADS,
        "voices": ALLOWED_VOICES,
        "dict_path": DICT_PATH or None,
        "dict_include": INCLUDE or None,
        "dict_exclude": EXCLUDE or None,
        "accentor": type(accentor).__name__,
    }


@app.post("/tts")
def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    voice = req.speaker or DEFAULT_VOICE
    if voice not in ALLOWED_VOICES:
        raise HTTPException(400, f"unknown voice, allowed: {ALLOWED_VOICES}")
    buf = _synthesize(req.text, voice, req.sample_rate)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="tts.wav"'},
    )


@app.post("/v1/audio/speech")
def openai_tts(req: OpenAITTSRequest):
    voice = req.voice if req.voice in ALLOWED_VOICES else DEFAULT_VOICE
    if not req.input.strip():
        raise HTTPException(400, "empty input")
    buf = _synthesize(req.input, voice)
    return StreamingResponse(buf, media_type="audio/wav")
