# tests/test_api.py
from fastapi.testclient import TestClient
import pytest
import pandas as pd
from api.main import app, scoring_system

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_data(tmp_path):
    """Setup test data before running tests."""
    mock_data = pd.DataFrame({
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
    
    csv_path = tmp_path / "test_candidates.csv"
    mock_data.to_csv(csv_path, index=False)
    
    scoring_system.load_data(str(csv_path))

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_score_candidates_endpoint():
    """Test the candidate scoring endpoint."""
    job_desc = {"description": "Senior Python Developer with AWS experience"}
    response = client.post("/score-candidates", json=job_desc)
    
    assert response.status_code == 200
    results = response.json()
    
    assert isinstance(results, list)
    assert len(results) <= 30
    
    for candidate in results:
        assert "name" in candidate
        assert "score" in candidate
        assert "relevant_experience" in candidate
        assert isinstance(candidate["score"], float)
        assert 0 <= candidate["score"] <= 100

def test_specific_role_matching():
    """Test scoring for specific roles."""
    # Test Python role
    python_response = client.post("/score-candidates", 
        json={"description": "Python Developer with AWS"})
    assert python_response.status_code == 200
    python_results = python_response.json()
    assert isinstance(python_results, list)
    assert len(python_results) >= 0
    
    # Test Ruby role
    ruby_response = client.post("/score-candidates", 
        json={"description": "Ruby on Rails Developer"})
    assert ruby_response.status_code == 200
    ruby_results = ruby_response.json()
    assert isinstance(ruby_results, list)
    assert len(ruby_results) >= 0

def test_long_description_validation():
    """Test validation of job description length."""
    long_desc = {"description": "x" * 201}
    response = client.post("/score-candidates", json=long_desc)
    
    assert response.status_code == 422
    error_detail = response.json()
    assert "at most 200 characters" in error_detail["detail"][0]["msg"]

def test_empty_description():
    """Test behavior with empty job description."""
    response = client.post("/score-candidates", json={"description": ""})
    
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) <= 30

@pytest.mark.parametrize("invalid_input", [
    {"wrong_field": "test"},
    {"description": 123},
    {},
    []
])
def test_invalid_input(invalid_input):
    """Test handling of various invalid inputs."""
    response = client.post("/score-candidates", json=invalid_input)
    assert response.status_code == 422