from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from src.dependencies.session import UserContext, verify_session_required
from src.services.bloom import detect_bloom, predict_bloom


router = APIRouter()


# ------------------------
# BLOOM DETECTION (past events)
# ------------------------


class BloomDetectionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the orchard location")
    longitude: float = Field(..., description="Longitude of the orchard location")


class BloomDetectionResponse(BaseModel):
    latitude: float
    longitude: float
    date_of_max_ebi: Optional[date] = None
    ebi_value: Optional[float] = None
    image_url: Optional[str] = None


@router.post("/bloom-detection", response_model=BloomDetectionResponse)
async def bloom_detection(
    request: BloomDetectionRequest,
    session: UserContext = Depends(verify_session_required),
):
    # Dummy implementation via service stub
    return await detect_bloom(request.latitude, request.longitude)


# ------------------------
# BLOOM PREDICTION (future events)
# ------------------------


class BloomPredictionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the orchard location")
    longitude: float = Field(..., description="Longitude of the orchard location")


class BloomPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    predicted_bloom_start: Optional[date] = None
    predicted_bloom_peak: Optional[date] = None
    confidence: Optional[float] = None


@router.post("/bloom-prediction", response_model=BloomPredictionResponse)
async def bloom_prediction(
    request: BloomPredictionRequest,
    session: UserContext = Depends(verify_session_required),
):
    # Dummy implementation via service stub
    return await predict_bloom(request.latitude, request.longitude)
