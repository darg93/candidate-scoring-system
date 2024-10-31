# api/models/schemas.py
from pydantic import BaseModel, Field, field_validator

class JobDescription(BaseModel):
    """Request model for job description input."""
    description: str = Field(..., max_length=200)

    @field_validator('description')
    @classmethod
    def validate_description_length(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("Job description must not exceed 200 characters")
        return v

class CandidateScore(BaseModel):
    """Response model for candidate scoring."""
    name: str
    score: float
    relevant_experience: str