#inference

from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


RAW_COLUMNS = [
    "Age",
    "Gender",
    "Family History",
    "Personal History",
    "Current Stressors",
    "Symptoms",
    "Severity",
    "Impact on Life",
    "Demographics",
    "Medical History",
    "Psychiatric History",
    "Substance Use",
    "Coping Mechanisms",
    "Social Support",
    "Lifestyle Factors",
]


def run_single(request_dict: dict, artifacts: dict) -> dict:
    t0 = time.perf_counter()

    df = pd.DataFrame([request_dict])

    # Convert API field names to training dataset column names
    df = df.rename(columns={
        "age": "Age",
        "gender": "Gender",
        "family_history": "Family History",
        "personal_history": "Personal History",
        "current_stressors": "Current Stressors",
        "symptoms": "Symptoms",
        "severity": "Severity",
        "impact_on_life": "Impact on Life",
        "demographics": "Demographics",
        "medical_history": "Medical History",
        "psychiatric_history": "Psychiatric History",
        "substance_use": "Substance Use",
        "coping_mechanisms": "Coping Mechanisms",
        "social_support": "Social Support",
        "lifestyle_factors": "Lifestyle Factors",
    })

    df = df[RAW_COLUMNS]

    probability, prediction = _score(df, artifacts)

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "single inference | pred=%d | prob=%.4f | latency=%.1fms",
        prediction[0],
        probability[0],
        latency_ms,
    )

    return {
        "prediction": int(prediction[0]),
        "probability": round(float(probability[0]), 6),
        "is_panic_disorder": bool(prediction[0] == 1),
    }


def run_batch(requests: list[dict], artifacts: dict) -> dict:
    t0 = time.perf_counter()

    df = pd.DataFrame(requests)

    df = df.rename(columns={
        "age": "Age",
        "gender": "Gender",
        "family_history": "Family History",
        "personal_history": "Personal History",
        "current_stressors": "Current Stressors",
        "symptoms": "Symptoms",
        "severity": "Severity",
        "impact_on_life": "Impact on Life",
        "demographics": "Demographics",
        "medical_history": "Medical History",
        "psychiatric_history": "Psychiatric History",
        "substance_use": "Substance Use",
        "coping_mechanisms": "Coping Mechanisms",
        "social_support": "Social Support",
        "lifestyle_factors": "Lifestyle Factors",
    })

    df = df[RAW_COLUMNS]

    probabilities, predictions = _score(df, artifacts)

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "batch inference | n=%d | panic_disorder_count=%d | latency=%.1fms",
        len(predictions),
        int(predictions.sum()),
        latency_ms,
    )

    return {
        "predictions": [int(p) for p in predictions],
        "probabilities": [
            round(float(p), 6) for p in probabilities
        ],
        "panic_disorder_count": int(predictions.sum()),
        "total": len(predictions),
    }


def _score(df: pd.DataFrame, artifacts: dict):
    preprocessor = artifacts["preprocessor"]
    feature_engineer = artifacts["feature_engineer"]
    model = artifacts["model"]

    # Apply the SAME fitted preprocessing used during training
    df_clean = preprocessor.transform(df)

    # Apply the SAME fitted feature engineering used during training
    df_eng = feature_engineer.transform(df_clean)

    # Align columns with the features used during model training
    expected = artifacts["feature_names"]

    for col in expected:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    df_eng = df_eng[expected]

    probabilities = model.predict_proba(df_eng)[:, 1]
    predictions = model.predict(df_eng)

    return probabilities, predictions