# api/services/scoring.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import List, Dict
from ..models.schemas import CandidateScore

class ScoringSystem:
    def __init__(self):
        self.tfidf = None
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.dt_model = DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
        
        self.best_model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.candidates_df = None
        self.feature_vectors = None

    def extract_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract numerical features from candidate data."""
        features = {
            'years_experience': 0,
            'senior_level': 0,
            'education_level': 0,
            'skill_count': 0
        }
        
        # Experience features
        exp_text = str(row['Experiences']).lower()
        if 'years' in exp_text:
            years = [int(s) for s in exp_text.split() if s.isdigit()]
            if years:
                features['years_experience'] = min(max(years), 20)
        
        # Seniority features
        features['senior_level'] = 1 if any(title in exp_text for title in 
            ['senior', 'lead', 'architect', 'manager']) else 0
            
        # Education features
        edu_text = str(row['Educations']).lower()
        if 'phd' in edu_text or 'doctorate' in edu_text:
            features['education_level'] = 3
        elif 'master' in edu_text:
            features['education_level'] = 2
        elif 'bachelor' in edu_text or 'degree' in edu_text:
            features['education_level'] = 1
            
        # Skills count
        skills_text = str(row['Skills']).lower()
        features['skill_count'] = len([s for s in skills_text.split(',') if s.strip()])
        
        return features

    def load_data(self, data_path: str) -> bool:
        """Load and preprocess candidate data."""
        try:
            # Load CSV file
            self.candidates_df = pd.read_csv(data_path)
            
            # Combine text fields for TF-IDF
            self.candidates_df['combined_text'] = self.candidates_df.apply(
                lambda x: ' '.join(filter(None, [
                    str(x['Skills'] or ''),
                    str(x['Experiences'] or ''),
                    str(x['Keywords'] or ''),
                    str(x['Summary'] or ''),
                    str(x['Educations'] or '')
                ])),
                axis=1
            )
            
            # Initialize and fit TF-IDF
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2),
                analyzer='word',
                min_df=1
            )
            
            # Get text features
            text_features = self.tfidf.fit_transform(
                self.candidates_df['combined_text'].fillna('')
            ).toarray()
            
            # Get numerical features
            numerical_features = []
            for _, row in self.candidates_df.iterrows():
                features = self.extract_features(row)
                numerical_features.append(list(features.values()))
            
            numerical_features = np.array(numerical_features)
            
            # Scale numerical features
            numerical_features_scaled = self.scaler.fit_transform(numerical_features)
            
            # Combine all features
            self.feature_vectors = np.hstack([
                text_features,
                numerical_features_scaled
            ])
            
            # Train the model with synthetic data
            self._train_model()
            
            self.is_fitted = True
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def _train_model(self):
        """Train the model with synthetic positive and negative examples."""
        # Create synthetic training data
        X_train = self.feature_vectors
        
        # Create synthetic labels (assuming higher feature values indicate better candidates)
        y_train = np.zeros(len(self.feature_vectors))
        # Mark top 30% as positive examples
        top_indices = np.argsort(np.mean(self.feature_vectors, axis=1))[-int(len(self.feature_vectors) * 0.3):]
        y_train[top_indices] = 1
        
        # Train both models
        self.rf_model.fit(X_train, y_train)
        self.dt_model.fit(X_train, y_train)
        
        # Select best model based on accuracy
        rf_score = accuracy_score(y_train, self.rf_model.predict(X_train))
        dt_score = accuracy_score(y_train, self.dt_model.predict(X_train))
        
        self.best_model = self.rf_model if rf_score >= dt_score else self.dt_model

    def score_candidates(self, job_description: str, top_n: int = 30) -> List[CandidateScore]:
        """Score candidates using combined approach of ML and similarity."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please load data first.")

        try:
            # Transform job description
            job_text_features = self.tfidf.transform([job_description]).toarray()
            
            # Calculate text similarity scores using cosine similarity
            text_similarities = cosine_similarity(
                job_text_features,
                self.feature_vectors[:, :-4]  # Exclude numerical features
            )[0]
            
            # Get model predictions
            ml_scores = self.best_model.predict_proba(self.feature_vectors)[:, 1]
            
            # Combine scores (70% ML, 30% similarity)
            final_scores = (0.7 * ml_scores) + (0.3 * text_similarities)
            
            # Get top candidates
            top_indices = np.argsort(final_scores)[-top_n:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                score = final_scores[idx]
                results.append(
                    CandidateScore(
                        name=f"{self.candidates_df.iloc[idx]['Name']}",
                        score=round(min(score * 100, 100), 2),
                        relevant_experience=str(self.candidates_df.iloc[idx]['Experiences'])[:1500]
                    )
                )
            
            return results
            
        except Exception as e:
            print(f"Error scoring candidates: {str(e)}")
            return []