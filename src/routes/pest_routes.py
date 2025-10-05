from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional
from src.dependencies.session import UserContext, verify_session_required
from src.services.pest import detect_pest


router = APIRouter()


class PestDetectionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the orchard location")
    longitude: float = Field(..., description="Longitude of the orchard location")


class PestDetectionResponse(BaseModel):
    detected_pest: bool
    confidence_score: Optional[float] = None
    image_url: Optional[str] = None


@router.post("/pest-detection", response_model=PestDetectionResponse)
async def pest_detection(
    request: PestDetectionRequest,
    session: UserContext = Depends(verify_session_required),
):
    # Dummy implementation via service stub
    return await detect_pest(request.latitude, request.longitude)

