# Candidate Scoring System

AI-powered candidate screening and scoring system built with FastAPI and scikit-learn.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the candidate database CSV file in the `data/` directory

4. Run the API:
```bash
uvicorn api.main:app --reload
```

5. Open frontend/index.html in your browser

## Project Structure

- `api/`: FastAPI backend application
- `data/`: Data files
- `frontend/`: Web interface
- `tests/`: Test files

## API Endpoints

- POST `/score-candidates`: Score candidates based on job description
- GET `/health`: Health check endpoint

## Testing

Run tests with:
```bash
pytest
```