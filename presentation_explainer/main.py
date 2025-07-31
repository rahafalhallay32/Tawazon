import os
import uuid
import time
import logging
import re
import asyncio
from typing import List, Tuple

import tempfile
import shutil
import aiofiles
import langid

import torch
import torchaudio
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# TTS imports
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from dto import (
    TTSResponse,
    HealthResponse
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agree to Coqui TOS
os.environ["COQUI_TOS_AGREED"] = "1"

# FastAPI app
app = FastAPI(
    title="XTTS Voice Cloning API",
    description="FastAPI service for XTTS voice cloning with Arabic and English support",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
config = None
supported_languages = [
    "en", "ar", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "zh-cn", "ja", "ko", "hu", "hi"
]

os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# 👇 YOUR LOCAL MODEL PATH
LOCAL_MODEL_PATH = "/Users/rahafalhallay/Desktop/xtts/xtts_v2"

async def load_model():
    """Load XTTS model on startup"""
    global model, config

    model_path = LOCAL_MODEL_PATH
    logger.info(f"Using model path: {model_path!r}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"XTTS model folder not found at {model_path!r}."
        )

    try:
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        logger.info("Loaded XTTS config.json")

        model_instance = Xtts.init_from_config(config)
        model_instance.load_checkpoint(
            config=config,
            checkpoint_dir=model_path,
            checkpoint_path=os.path.join(model_path, "model.pth"),
            speaker_file_path=os.path.join(model_path, "speakers_xtts.pth"),
            vocab_path=os.path.join(model_path, "vocab.json"),
            eval=True,
        )

        if torch.cuda.is_available():
            model_instance.cuda()
            logger.info("XTTS model moved to GPU")
        else:
            logger.info("XTTS model loaded on CPU")

        model_instance.eval()
        model = model_instance
        app.state.tts_model = model
        logger.info("XTTS model is ready")

    except Exception as e:
        logger.error(f"Failed to load XTTS model: {e}", exc_info=True)
        raise

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
async def frontend(request: Request):
    user_name = "Rahaf"  # Example dynamic variable
    return templates.TemplateResponse("index.html", {"request": request, "user_name": user_name})


@app.on_event("startup")
async def startup_event():
    await load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        xtts_model_loaded=model is not None,
        supported_languages=supported_languages,
        gpu_available=torch.cuda.is_available()
    )

def split_text_into_sentences(text: str, max_chars: int = 200) -> list[str]:
    sentence_boundaries = r'[.!?؟،؛]'
    sentences = re.split(f'({sentence_boundaries})', text)

    processed_sentences = []
    current_sentence = ""

    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            current_sentence = sentences[i] + sentences[i + 1]
        else:
            current_sentence = sentences[i]

        while len(current_sentence) > max_chars:
            last_space = current_sentence[:max_chars].rfind(' ')
            if last_space != -1:
                processed_sentences.append(current_sentence[:last_space].strip())
                current_sentence = current_sentence[last_space:].strip()
            else:
                processed_sentences.append(current_sentence[:max_chars])
                current_sentence = current_sentence[max_chars:].strip()

        if current_sentence:
            processed_sentences.append(current_sentence.strip())

    return [s for s in processed_sentences if s.strip()]

async def process_single_sentence(
    sentence: str,
    language: str,
    gpt_cond_latent,
    speaker_embedding,
    repetition_penalty: float,
    temperature: float,
    index: int
) -> Tuple[int, np.ndarray]:
    logger.info(f"Processing sentence {index}: {sentence}")
    start = time.time()

    out = model.inference(
        sentence,
        language,
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
    )

    logger.info(f"Completed sentence {index} in {time.time() - start:.2f}s")
    return index, out["wav"]

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: str = Form("ar"),
    temperature: float = Form(0.75),
    repetition_penalty: float = Form(5.0),
    reference_audio: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if language not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

    if len(text.strip()) < 2:
        raise HTTPException(status_code=400, detail="Text too short")

    sentences = split_text_into_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences")

    task_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix=f"tts_{task_id}_")

    try:
        ref_audio_path = os.path.join(temp_dir, "reference.wav")
        async with aiofiles.open(ref_audio_path, "wb") as f:
            await f.write(await reference_audio.read())

        detected_lang = langid.classify(text)[0].strip()
        output_filename = f"output_{task_id}.wav"
        output_path = os.path.join("outputs", output_filename)

        logger.info("Extracting voice characteristics...")
        cond_start = time.time()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=ref_audio_path,
            max_ref_length=60,
            gpt_cond_len=30,
            gpt_cond_chunk_len=4
        )
        cond_time = time.time() - cond_start

        logger.info(f"Processing {len(sentences)} sentences...")
        inference_start = time.time()
        tasks = [
            process_single_sentence(
                sentence,
                language,
                gpt_cond_latent,
                speaker_embedding,
                repetition_penalty,
                temperature,
                i
            )
            for i, sentence in enumerate(sentences)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        all_wavs = [wav for _, wav in results]
        inference_time = time.time() - inference_start

        final_wav = np.concatenate(all_wavs)
        torchaudio.save(output_path, torch.tensor(final_wav).unsqueeze(0), 24000)

        total_time = time.time() - cond_start
        duration = len(final_wav) / 24000
        rtf = inference_time / duration if duration > 0 else 0

        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        return TTSResponse(
            success=True,
            message="Speech synthesized successfully",
            audio_url=f"/download/{output_filename}",
            metrics={
                "conditioning_time_ms": round(cond_time * 1000),
                "inference_time_ms": round(inference_time * 1000),
                "total_time_ms": round(total_time * 1000),
                "audio_duration_s": round(duration, 2),
                "real_time_factor": round(rtf, 2),
                "detected_language": detected_lang,
                "model_language": language,
                "text_length": len(text),
                "audio_sample_rate": 24000,
                "number_of_sentences": len(sentences),
                "parallel_processing": True
            },
            task_id=task_id
        )

    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.get("/download/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/languages")
async def get_supported_languages():
    return {"supported_languages": supported_languages, "total_count": len(supported_languages)}

@app.delete("/cleanup")
async def cleanup_outputs():
    deleted = 0
    try:
        for f in os.listdir("outputs"):
            path = os.path.join("outputs", f)
            if os.path.isfile(path) and time.time() - os.path.getctime(path) > 3600:
                os.remove(path)
                deleted += 1
        return {"message": f"Deleted {deleted} files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

async def cleanup_temp_dir(temp_dir: str):
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp dir: {temp_dir}")
    except Exception as e:
        logger.error(f"Temp cleanup error: {str(e)}")
