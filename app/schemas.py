#schemas
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PanicDisorderRequest(BaseModel):

    age: int = Field(...,description="Age of the participant",)
    gender: str = Field(...,description="Gender of the participant",)
    family_history: str = Field(...,description="Family history",)
    personal_history: str = Field(...,description="Personal history",)
    current_stressors: str = Field(...,description="Current stressors",)
    symptoms: str = Field(...,description="Reported symptoms",)
    severity: str = Field(...,description="Symptom severity",)
    impact_on_life: str = Field(...,description="Impact on life",)
    demographics: str = Field(...,description="Demographic information",)
    medical_history: str = Field(...,description="Medical history",)
    psychiatric_history: str = Field(...,description="Psychiatric history",)
    substance_use: str = Field(...,description="Substance use",)
    coping_mechanisms: str = Field(...,description="Coping mechanisms",)
    social_support: str = Field(...,description="Social support",)
    lifestyle_factors: str = Field(...,description="Lifestyle factors",)
    
    model_config = { 
                    "json_schema_extra": { 
                        "example": { 
                            "age": 34, 
                            "gender": "Female", 
                            "family_history": "Yes", 
                            "personal_history": "No", 
                            "current_stressors": "Low", 
                            "symptoms": "Chest pain", 
                            "severity": "Moderate", 
                            "impact_on_life": "Moderate", 
                            "demographics": "Urban", 
                            "medical_history": "Asthma", 
                            "psychiatric_history": "Bipolar disorder", 
                            "substance_use": "Drugs", 
                            "coping_mechanisms": "Exercise", 
                            "social_support": "Low", 
                            "lifestyle_factors": "Diet", 
                            } 
                        } 
                    }


class PredictionResponse(BaseModel):
    prediction: int = Field(...,description="Predicted class: 0 = No Panic Disorder, 1 = Panic Disorder",)
    probability: float = Field(...,ge=0.0,le=1.0,description="Predicted probability of Panic Disorder",)
    is_panic_disorder: bool = Field(...,description="True when the prediction is Panic Disorder",)


class BatchPanicDisorderRequest(BaseModel):
    requests: list[PanicDisorderRequest]

class BatchPredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    panic_disorder_count: int
    total: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
class ModelInfoResponse(BaseModel):
    metrics: dict[str, Any]
    feature_count: int
    features: list[str]
