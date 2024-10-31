# api/utils/data_loader.py
import os

def get_data_path() -> str:
    """Get the path to the candidates data file."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'ZIPDEV_-_Candidate_Database_Code_challenge.xlsx - Sheet1.csv'
    )