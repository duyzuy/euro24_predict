#!/usr/bin/env python3
"""
Euro 2024 Prediction API
========================

FastAPI application to serve Euro 2024 match prediction models.
Provides REST endpoints for predicting match results and goals.

Features:
- Match result prediction (1X2)
- Goals prediction (total goals)
- Team statistics retrieval
- Model health checks
- Batch predictions
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Euro 2024 Prediction API",
    description="AI-powered predictions for Euro 2024 football matches",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
result_model = None
goals_model = None
feature_names = None
team_stats = None

# Pydantic models for request/response
class MatchRequest(BaseModel):
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    
class BatchMatchRequest(BaseModel):
    matches: List[MatchRequest] = Field(..., description="List of matches to predict")

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str = Field(..., description="1=Home Win, X=Draw, 2=Away Win")
    predicted_goals: float
    confidence: float
    probabilities: Dict[str, float] = Field(..., description="Win/Draw/Away probabilities")
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class TeamStatsResponse(BaseModel):
    team_name: str
    stats: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    data_loaded: bool
    timestamp: str

class PredictionService:
    """Service class for handling predictions"""
    
    def __init__(self, data_path: str = "./data/"):
        self.data_path = data_path
        self.result_model = None
        self.goals_model = None
        self.feature_names = None
        self.team_stats = None
        
    def load_models(self, model_version: str = "v2_enhanced"):
        """Load trained models and data"""
        try:
            # Load models
            result_filename = f"{self.data_path}euro24_results_{model_version}.joblib"
            goals_filename = f"{self.data_path}euro24_goals_{model_version}.joblib"
            feature_filename = f"{self.data_path}feature_names_{model_version}.joblib"
            
            self.result_model = joblib.load(result_filename)
            self.goals_model = joblib.load(goals_filename)
            self.feature_names = joblib.load(feature_filename)
            
            # Load team statistics
            self.team_stats = pd.read_csv(f'{self.data_path}euro24_qualifiers_norm.csv', index_col=0)
            
            logger.info(f"Successfully loaded {model_version} models and data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Try to load original models as fallback
            try:
                result_filename = f"{self.data_path}euro20_results.joblib"
                goals_filename = f"{self.data_path}euro20_goals.joblib"
                
                self.result_model = joblib.load(result_filename)
                self.goals_model = joblib.load(goals_filename)
                self.team_stats = pd.read_csv(f'{self.data_path}euro24_qualifiers_norm.csv', index_col=0)
                
                logger.info("Loaded original models as fallback")
                return True
                
            except Exception as e2:
                logger.error(f"Failed to load any models: {str(e2)}")
                return False
    
    def create_match_features(self, home_team: str, away_team: str):
        """Create feature vector for a single match"""
        if self.team_stats is None:
            raise ValueError("Team statistics not loaded")
            
        # Get team statistics
        home_stats = self.team_stats[self.team_stats['team_name'] == home_team].copy()
        away_stats = self.team_stats[self.team_stats['team_name'] == away_team].copy()
        
        if len(home_stats) == 0:
            raise ValueError(f"No statistics found for home team: {home_team}")
        if len(away_stats) == 0:
            raise ValueError(f"No statistics found for away team: {away_team}")
        
        # Remove team_name column and add suffixes
        home_stats = home_stats.drop('team_name', axis=1).add_suffix('_home')
        away_stats = away_stats.drop('team_name', axis=1).add_suffix('_away')
        
        # Combine base features
        match_features = []
        match_features.extend(home_stats.values[0])
        match_features.extend(away_stats.values[0])
        
        # Add enhanced features if using enhanced model
        if self.feature_names and 'home_attack_strength' in self.feature_names:
            home_attack_strength = home_stats['goals_home'].values[0] / home_stats['attempts_home'].values[0] if home_stats['attempts_home'].values[0] > 0 else 0
            away_attack_strength = away_stats['goals_away'].values[0] / away_stats['attempts_away'].values[0] if away_stats['attempts_away'].values[0] > 0 else 0
            home_defense_strength = 1 - (home_stats['goals_conceded_home'].values[0] / 10)
            away_defense_strength = 1 - (away_stats['goals_conceded_away'].values[0] / 10)
            
            match_features.extend([
                home_attack_strength, away_attack_strength,
                home_defense_strength, away_defense_strength,
                home_attack_strength - away_defense_strength,
                away_attack_strength - home_defense_strength
            ])
        
        # Create DataFrame with proper feature names
        if self.feature_names:
            features_df = pd.DataFrame([match_features], columns=self.feature_names)
        else:
            # Fallback for original model
            home_cols = list(home_stats.columns)
            away_cols = list(away_stats.columns)
            features_df = pd.DataFrame([match_features], columns=home_cols + away_cols)
        
        return features_df
    
    def predict_match(self, home_team: str, away_team: str) -> PredictionResponse:
        """Predict a single match"""
        if not self.result_model or not self.goals_model:
            raise ValueError("Models not loaded")
        
        try:
            # Create features
            features = self.create_match_features(home_team, away_team)
            
            # Make predictions
            result_pred = self.result_model.predict(features)[0]
            result_proba = self.result_model.predict_proba(features)[0]
            goals_pred = self.goals_model.predict(features)[0]
            
            # Decode result prediction
            result_label = 'X' if result_pred == 0 else str(result_pred)
            
            # Create probabilities dictionary
            prob_labels = ['Draw', 'Home Win', 'Away Win']
            probabilities = {}
            for i, label in enumerate(prob_labels):
                if i < len(result_proba):
                    probabilities[label] = float(result_proba[i])
                else:
                    probabilities[label] = 0.0
            
            # Calculate confidence
            confidence = float(np.max(result_proba))
            
            return PredictionResponse(
                home_team=home_team,
                away_team=away_team,
                predicted_result=result_label,
                predicted_goals=float(goals_pred),
                confidence=confidence,
                probabilities=probabilities
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {home_team} vs {away_team}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Initialize prediction service
prediction_service = PredictionService()

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    logger.info("Starting Euro 2024 Prediction API...")
    success = prediction_service.load_models()
    if not success:
        logger.error("Failed to load models on startup")
    else:
        logger.info("API ready to serve predictions!")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Euro 2024 Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = prediction_service.result_model is not None and prediction_service.goals_model is not None
    data_loaded = prediction_service.team_stats is not None
    
    return HealthResponse(
        status="healthy" if models_loaded and data_loaded else "unhealthy",
        models_loaded=models_loaded,
        data_loaded=data_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.get("/teams", response_model=List[str])
async def get_teams():
    """Get list of available teams"""
    if prediction_service.team_stats is None:
        raise HTTPException(status_code=500, detail="Team data not loaded")
    
    teams = prediction_service.team_stats['team_name'].tolist()
    return sorted(teams)

@app.get("/teams/{team_name}/stats", response_model=TeamStatsResponse)
async def get_team_stats(team_name: str):
    """Get statistics for a specific team"""
    if prediction_service.team_stats is None:
        raise HTTPException(status_code=500, detail="Team data not loaded")
    
    team_data = prediction_service.team_stats[prediction_service.team_stats['team_name'] == team_name]
    
    if len(team_data) == 0:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    
    # Convert to dictionary, excluding team_name
    stats_dict = team_data.drop('team_name', axis=1).iloc[0].to_dict()
    
    return TeamStatsResponse(
        team_name=team_name,
        stats={k: float(v) for k, v in stats_dict.items()}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(match: MatchRequest):
    """Predict a single match result"""
    return prediction_service.predict_match(match.home_team, match.away_team)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_matches(request: BatchMatchRequest):
    """Predict multiple matches"""
    predictions = []
    errors = []
    
    for match in request.matches:
        try:
            prediction = prediction_service.predict_match(match.home_team, match.away_team)
            predictions.append(prediction)
        except Exception as e:
            errors.append(f"{match.home_team} vs {match.away_team}: {str(e)}")
    
    # Calculate summary statistics
    if predictions:
        avg_goals = sum(p.predicted_goals for p in predictions) / len(predictions)
        high_confidence = len([p for p in predictions if p.confidence >= 0.6])
        over_2_5 = len([p for p in predictions if p.predicted_goals > 2.5])
        
        summary = {
            "total_predictions": len(predictions),
            "successful_predictions": len(predictions),
            "failed_predictions": len(errors),
            "average_goals": round(avg_goals, 2),
            "high_confidence_predictions": high_confidence,
            "over_2_5_goals": over_2_5,
            "errors": errors if errors else None
        }
    else:
        summary = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": len(errors),
            "errors": errors
        }
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary
    )

@app.get("/predict/{home_team}/vs/{away_team}", response_model=PredictionResponse)
async def predict_match_get(
    home_team: str, 
    away_team: str,
    format: Optional[str] = Query(None, description="Response format (json)")
):
    """Predict match via GET request (URL parameters)"""
    return prediction_service.predict_match(home_team, away_team)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )