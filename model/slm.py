"""
SLM (Small Language Model) integration — loads and runs the model
for natural language response generation.

Uses Qwen2.5-3B-Instruct by default (≤3B params, open weights).
Falls back to template-based response if model loading fails.
"""

import os
import asyncio
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Model configuration — can be overridden via environment variables
MODEL_ID = os.getenv("SLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("SLM_MAX_TOKENS", "1024"))
DEVICE = os.getenv("SLM_DEVICE", "auto")  # "cpu", "cuda", "mps", "auto"
USE_4BIT = os.getenv("SLM_USE_4BIT", "true").lower() == "true"

# Global model and tokenizer (loaded lazily)
_model = None
_tokenizer = None
_model_loaded = False
_loading_lock = asyncio.Lock()


def _load_model():
    """Load the SLM model and tokenizer (synchronous, called once)."""
    global _model, _tokenizer, _model_loaded

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading SLM: {MODEL_ID} ...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        # Configure quantization for memory efficiency
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if USE_4BIT:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit quantization")
            except Exception:
                logger.warning("bitsandbytes not available, loading in float16")
                load_kwargs["device_map"] = DEVICE

        if "quantization_config" not in load_kwargs:
            load_kwargs["device_map"] = DEVICE

        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        _model.eval()
        _model_loaded = True
        logger.info(f"SLM loaded successfully: {MODEL_ID}")

    except Exception as e:
        logger.error(f"Failed to load SLM: {e}")
        _model_loaded = False


async def ensure_model_loaded():
    """Ensure model is loaded (async-safe, idempotent)."""
    global _model_loaded
    async with _loading_lock:
        if not _model_loaded:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _load_model)


def is_model_loaded() -> bool:
    return _model_loaded


SYSTEM_PROMPT = """You are GeoAI Outdoor Assistant, an AI that recommends the best nearby places and times for outdoor activities.

You receive structured data about weather, air quality, geospatial context, and scored recommendations. Your job is to:

1. Give a clear, specific recommendation: which place and when.
2. Explain WHY this is the best option based on the data.
3. Provide practical advice: what to carry, what to watch for.
4. Mention when to AVOID going and why.
5. Account for any personal health conditions mentioned.
6. Keep responses concise but informative.
7. Always cite the actual data values (temperature, AQI, wind, etc.).
8. Never make up data — only use what is provided in the context.

Format your response in a clear, readable way with sections for:
- Best Recommendation
- Weather & Conditions
- What to Carry / Prepare
- When to Avoid
- Alternative Options (if available)"""


def _build_messages(user_query: str, context: str) -> list[dict]:
    """Build chat messages for the SLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Based on the following live data, answer the user's question.

USER QUESTION: {user_query}

LIVE DATA AND ANALYSIS:
{context}

Please provide a clear, grounded recommendation based ONLY on the data above. Do not invent any weather values, places, or data.""",
        },
    ]


async def generate_response(user_query: str, context: str) -> str:
    """
    Generate a natural language response using the SLM.

    Args:
        user_query: The user's original free-form question.
        context: Structured data context with scores, weather, AQI, etc.

    Returns:
        str: The SLM-generated response.
    """
    await ensure_model_loaded()

    if not _model_loaded or _model is None or _tokenizer is None:
        raise RuntimeError("SLM not loaded")

    messages = _build_messages(user_query, context)

    def _generate():
        import torch

        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = _tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate)


async def get_model_info() -> dict:
    """Return info about the currently loaded model."""
    return {
        "model_id": MODEL_ID,
        "loaded": _model_loaded,
        "max_tokens": MAX_NEW_TOKENS,
        "device": DEVICE,
        "quantization": "4-bit" if USE_4BIT else "float16",
    }
