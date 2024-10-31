# api/models/__init__.py
"""
Models package containing Pydantic schemas and data models.
"""

from .schemas import JobDescription, CandidateScore

__all__ = ['JobDescription', 'CandidateScore']