#main
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.loader import get_artifacts
from app.inference import run_batch, run_single
from app.schemas import (
    BatchPanicDisorderRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PanicDisorderRequest,
    PredictionResponse,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts when the FastAPI application starts.
    """

    log.info("startup.begin")

    try:
        get_artifacts()
        log.info("startup.complete — API ready")

    except Exception as e:
        log.error("startup.failed: %s", e)
        raise

    yield

    log.info("shutdown")


# --------------- FASTAPI APPLICATION

app = FastAPI(
    title="Panic Disorder Detection API",
    description=(
        "Machine-learning inference API for panic disorder prediction "
        "using demographic, psychological, behavioral, and clinical predictors."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

################## CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log HTTP method, endpoint, status code, and request latency.
    """

    t0 = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - t0) * 1000

    log.info(
        "http.request | method=%s | path=%s | status=%s | latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )

    return response


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
)
async def health():
    """
    Check whether the model artifacts are available.
    """

    try:
        arts = get_artifacts()

        return HealthResponse(
            status="ok",
            model_loaded=True,
        )

    except Exception as e:
        log.error("health.check.failed: %s", e)

        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {e}",
        )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["ops"],
)
async def model_info():
    """
    Return information about the loaded model and evaluation metrics.
    """

    try:
        arts = get_artifacts()

        return ModelInfoResponse(
            metrics=arts["metrics"],
            feature_count=len(arts["feature_names"]),
            features=arts["feature_names"],
        )

    except Exception as e:
        log.error("model.info.failed: %s", e)

        raise HTTPException(
            status_code=503,
            detail=f"Model information unavailable: {e}",
        )


# =============================================================================
# SINGLE PREDICTION
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
)
async def predict(request: PanicDisorderRequest):
    """
    Generate a panic disorder prediction for one patient record.
    """

    try:
        arts = get_artifacts()

        t0 = time.perf_counter()

        result = run_single(
            request.model_dump(),
            arts,
        )

        latency = time.perf_counter() - t0

        log.info(
            "predict.single | pred=%s | prob=%.4f | latency_ms=%.2f",
            result["prediction"],
            result["probability"],
            latency * 1000,
        )

        return PredictionResponse(**result)

    except Exception as e:
        log.error("predict.single.error: %s", e)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


# =============================================================================
# BATCH PREDICTION
# =============================================================================

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
)
async def predict_batch(
    body: BatchPanicDisorderRequest,
):
    """
    Generate panic disorder predictions for multiple records.
    """

    if len(body.requests) == 0:
        raise HTTPException(
            status_code=422,
            detail="Empty batch.",
        )

    if len(body.requests) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Maximum batch size is 1000.",
        )

    try:
        arts = get_artifacts()

        t0 = time.perf_counter()

        requests_dicts = [
            request.model_dump()
            for request in body.requests
        ]

        result = run_batch(
            requests_dicts,
            arts,
        )

        latency = time.perf_counter() - t0

        log.info(
            "predict.batch | total=%s | panic_disorder=%s | latency_ms=%.2f",
            result["total"],
            result["panic_disorder_count"],
            latency * 1000,
        )

        return BatchPredictionResponse(**result)

    except Exception as e:
        log.error("predict.batch.error: %s", e)

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        )
