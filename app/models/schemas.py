"""Pydantic models for request/response schemas."""

from __future__ import annotations

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ActionUnit(BaseModel):
    Name: str
    Value: float


class FacialExpressions(BaseModel):
    DominantBasicEmotion: str
    BasicEmotions: Dict[str, float]
    Valence: float = Field(ge=0, le=1)
    Arousal: float = Field(ge=0, le=1)


class Characteristics(BaseModel):
    Gender: str = "Unknown"
    Age: str = "Unknown"
    Glasses: str = "Unknown"
    Moustache: str = "None"
    Beard: str = "None"


class EmotionResponse(BaseModel):
    """Response model for emotion analysis."""
    
    FaceAnalyzed: bool
    FacialExpressions: Optional[FacialExpressions] = None
    Characteristics: Optional[Characteristics] = None
    ActionUnits: List[ActionUnit] = Field(default_factory=list)
    Confidence: float = 0.0
    HeadOrientation: List[float] = Field(default_factory=list)
    BoundingBox: List[int] = Field(default_factory=list)
    NumberOfFaces: int = 0
    Focus: float = 0.0
    ActionableEmotion: Optional[str] = None
    RotationApplied: int = 0
    ProcessingTimeMs: Optional[float] = None
    Error: Optional[str] = None
    
    def model_post_init(self, __context: dict) -> None:
        """Set default values after initialization."""
        if self.Characteristics is None:
            self.Characteristics = Characteristics()
        if self.HeadOrientation == []:
            self.HeadOrientation = [0.0, 0.0, 0.0]
        if self.BoundingBox == []:
            self.BoundingBox = [0, 0, 0, 0]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class BatchResponse(BaseModel):
    results: List[EmotionResponse]
    count: int
