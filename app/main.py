"""
GeoAI Outdoor Assistant — FastAPI application.

HTTP API that takes free-form questions about outdoor activities
and returns grounded recommendations with weather, AQI, and
geospatial context.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional

from engine.orchestrator import process_query
from model.slm import get_model_info, ensure_model_loaded, is_model_loaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Whether to use SLM (set to false for faster dev/testing without GPU)
USE_SLM = os.getenv("USE_SLM", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload the SLM model."""
    if USE_SLM:
        logger.info("Preloading SLM model in background...")
        try:
            await ensure_model_loaded()
            logger.info("SLM model loaded ✓")
        except Exception as e:
            logger.warning(f"SLM model load failed: {e}. Will use fallback responses.")
    else:
        logger.info("SLM disabled (USE_SLM=false). Using template responses.")
    yield


app = FastAPI(
    title="GeoAI Outdoor Assistant",
    description=(
        "AI assistant that recommends the best nearby places and times "
        "for outdoor activities, powered by a Small Language Model (≤3B params) "
        "and live open data sources."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates for the web UI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")
STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Free-form question about an outdoor activity",
        examples=[
            "Where can I go running tomorrow evening near Koregaon Park?",
            "Best time this weekend for a kids' cricket match within 15 km of Whitefield?",
            "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
        ],
    )
    use_slm: Optional[bool] = Field(
        default=None,
        description="Force SLM on/off for this request. Defaults to server setting.",
    )
    user_lat: Optional[float] = Field(
        default=None,
        description="User's current GPS latitude (optional, for precise local distance)",
    )
    user_lon: Optional[float] = Field(
        default=None,
        description="User's current GPS longitude (optional, for precise local distance)",
    )


class RecommendationItem(BaseModel):
    place_name: str
    place_type: str
    distance_km: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    time_slot: str
    composite_score: float
    label: str
    weather_summary: dict
    notes: list[str]
    travel: Optional[dict] = None


class QueryResponse(BaseModel):
    query: str
    parsed: dict
    location: Optional[dict] = None
    recommendations: list[RecommendationItem]
    avoid: list[dict] = []
    response_text: str
    data_sources: list[str]
    model_info: dict
    processing_time_seconds: float


# ── API Endpoints ─────────────────────────────────────────────────

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Main endpoint: send a free-form outdoor activity question
    and get a grounded recommendation.
    """
    start = time.time()

    use_slm_flag = request.use_slm if request.use_slm is not None else USE_SLM

    try:
        result = await process_query(
            request.query,
            use_slm=use_slm_flag,
            user_lat=request.user_lat,
            user_lon=request.user_lon,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    model_info = await get_model_info()
    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        query=request.query,
        parsed=result["parsed_query"],
        location=result.get("location"),
        recommendations=result.get("recommendations", []),
        avoid=result.get("avoid", []),
        response_text=result["response_text"],
        data_sources=result.get("data_sources", []),
        model_info=model_info,
        processing_time_seconds=elapsed,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model-info")
async def model_info():
    """Return info about the loaded SLM."""
    return await get_model_info()


@app.get("/download-ppt")
async def download_ppt():
    """Download the 8-slide presentation (.pptx)."""
    candidates = [
        os.path.join(os.path.dirname(BASE_DIR), "GeoAI_Outdoor_Assistant_Presentation.pptx"),
        "/Users/pratyush/.gemini/antigravity/brain/0c6e9a79-1ff5-4aaf-835d-01dac5828376/GeoAI_Outdoor_Assistant_Presentation.pptx",
        os.path.join(BASE_DIR, "GeoAI_Outdoor_Assistant_Presentation.pptx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return FileResponse(
                path=p,
                filename="GeoAI_Outdoor_Assistant_Presentation.pptx",
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
    raise HTTPException(status_code=404, detail="Presentation PowerPoint file not found")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception:
        return HTMLResponse(
            content="<h1>GeoAI Outdoor Assistant API</h1>"
            "<p>POST to /ask with {\"query\": \"your question\"}</p>"
            "<p>See /docs for full API documentation.</p>",
            status_code=200,
        )
