# tests/test_scoring.py
import pytest
import pandas as pd
import numpy as np
from api.services.scoring import ScoringSystem
from api.models.schemas import CandidateScore

@pytest.fixture
def mock_data():
    """Create mock candidate data for testing."""
    return pd.DataFrame({
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Skills': [
            'python, machine learning, aws',
            'ruby, postgresql, react',
            'golang, docker, kubernetes'
        ],
        'Experiences': [
            '5 years Senior Python Developer at Tech Corp',
            '3 years Full Stack Developer with Ruby at StartUp Inc',
            '4 years Cloud Engineer with Go at Big Tech'
        ],
        'Keywords': ['python, aws', 'ruby, rails', 'golang, cloud'],
        'Summary': [
            'Experienced python developer',
            'Ruby on Rails expert',
            'Golang specialist'
        ],
        'Educations': [
            'Bachelor in Computer Science',
            'Master in Software Engineering',
            'PhD in Computer Engineering'
        ]
    })

@pytest.fixture
def scoring_system(mock_data, tmp_path):
    """Initialize scoring system with mock data."""
    csv_path = tmp_path / "test_candidates.csv"
    mock_data.to_csv(csv_path, index=False)
    
    system = ScoringSystem()
    system.load_data(str(csv_path))
    return system

def test_scoring_system_initialization(scoring_system):
    """Test if scoring system initializes correctly."""
    assert scoring_system.candidates_df is not None
    assert scoring_system.feature_vectors is not None
    assert scoring_system.tfidf is not None
    assert scoring_system.is_fitted is True
    assert scoring_system.best_model is not None

def test_experience_feature_extraction(scoring_system):
    """Test the experience feature extraction functionality."""
    test_row = pd.Series({
        'Experiences': '5 years Senior Python Developer',
        'Skills': 'python, django, flask',
        'Educations': 'Bachelor in Computer Science',
        'Keywords': '',
        'Summary': ''
    })
    
    features = scoring_system.extract_features(test_row)
    
    assert isinstance(features, dict)
    assert features['years_experience'] == 5
    assert features['senior_level'] == 1
    assert features['education_level'] == 1
    assert features['skill_count'] == 3

def test_candidate_scoring(scoring_system):
    """Test candidate scoring functionality."""
    results = scoring_system.score_candidates(
        "Senior Python Developer with AWS experience"
    )
    
    assert isinstance(results, list)
    assert len(results) >= 0
    assert all(isinstance(r, CandidateScore) for r in results)
    assert all(0 <= r.score <= 100 for r in results)

def test_scoring_with_empty_description(scoring_system):
    """Test scoring behavior with empty job description."""
    results = scoring_system.score_candidates("")
    assert isinstance(results, list)
    assert len(results) <= 30

def test_scoring_with_irrelevant_description(scoring_system):
    """Test scoring behavior with irrelevant job description."""
    results = scoring_system.score_candidates("underwater basket weaving expert")
    assert isinstance(results, list)
    assert len(results) <= 30

def test_error_handling(tmp_path):
    """Test error handling in data loading and scoring."""
    system = ScoringSystem()
    
    # Test with non-existent file
    assert not system.load_data(str(tmp_path / "nonexistent.csv"))
    
    # Test scoring without fitted model
    with pytest.raises(ValueError, match="Model is not fitted"):
        system.score_candidates("test")

def test_education_level_detection():
    """Test education level detection."""
    system = ScoringSystem()
    
    # Test PhD detection
    phd_row = pd.Series({
        'Educations': 'PhD in Computer Science',
        'Experiences': '',
        'Skills': '',
        'Keywords': '',
        'Summary': ''
    })
    assert system.extract_features(phd_row)['education_level'] == 3
    
    # Test Master detection
    master_row = pd.Series({
        'Educations': 'Master in Engineering',
        'Experiences': '',
        'Skills': '',
        'Keywords': '',
        'Summary': ''
    })
    assert system.extract_features(master_row)['education_level'] == 2
    
    # Test Bachelor detection
    bachelor_row = pd.Series({
        'Educations': 'Bachelor in Science',
        'Experiences': '',
        'Skills': '',
        'Keywords': '',
        'Summary': ''
    })
    assert system.extract_features(bachelor_row)['education_level'] == 1

def test_model_consistency(scoring_system):
    """Test consistency of model predictions."""
    job_desc = "Senior Python Developer"
    results1 = scoring_system.score_candidates(job_desc)
    results2 = scoring_system.score_candidates(job_desc)
    
    # Compare scores for consistency
    assert len(results1) == len(results2)
    if len(results1) > 0:
        assert results1[0].score == results2[0].score

def test_feature_vectors_shape(scoring_system):
    """Test the shape of feature vectors."""
    # Get actual feature vector size
    actual_feature_size = scoring_system.feature_vectors.shape[1]
    
    # Verify it's a reasonable size (should be text features + numerical features)
    assert actual_feature_size > 0
    assert actual_feature_size == len(scoring_system.tfidf.get_feature_names_out()) + 4