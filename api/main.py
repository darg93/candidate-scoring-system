# api/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from .models.schemas import JobDescription, CandidateScore
from .services.scoring import ScoringSystem
from .utils.data_loader import get_data_path
from typing import List
from pydantic import ValidationError

# Initialize scoring system
scoring_system = ScoringSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    if not scoring_system.load_data(get_data_path()):
        raise RuntimeError("Failed to load candidate data")
    yield
    # Shutdown
    # Add any cleanup here if needed

app = FastAPI(
    title="Candidate Scoring API",
    lifespan=lifespan
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/score-candidates", response_model=List[CandidateScore])
async def score_candidates(job_desc: JobDescription) -> List[CandidateScore]:
    """
    Score candidates based on job description.
    Returns top 30 candidates with their scores.
    """
    try:
        return scoring_system.score_candidates(job_desc.description)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))