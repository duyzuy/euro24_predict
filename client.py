#!/usr/bin/env python3
"""
Euro 2024 Prediction API Client
===============================

Example client to demonstrate how to use the FastAPI prediction service.
Shows various ways to interact with the API endpoints.
"""

import requests
import json
from typing import List, Dict

class Euro2024Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_teams(self) -> List[str]:
        """Get list of available teams"""
        response = requests.get(f"{self.base_url}/teams")
        return response.json()
    
    def get_team_stats(self, team_name: str) -> Dict:
        """Get statistics for a specific team"""
        response = requests.get(f"{self.base_url}/teams/{team_name}/stats")
        return response.json()
    
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Predict a single match using POST"""
        data = {
            "home_team": home_team,
            "away_team": away_team
        }
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()
    
    def predict_match_get(self, home_team: str, away_team: str) -> Dict:
        """Predict a single match using GET"""
        response = requests.get(f"{self.base_url}/predict/{home_team}/vs/{away_team}")
        return response.json()
    
    def predict_batch(self, matches: List[Dict[str, str]]) -> Dict:
        """Predict multiple matches"""
        data = {"matches": matches}
        response = requests.post(f"{self.base_url}/predict/batch", json=data)
        return response.json()

def main():
    """Demonstrate API usage"""
    client = Euro2024Client()
    
    print("🚀 Euro 2024 Prediction API Client Demo")
    print("=" * 50)
    
    # Health check
    print("\n1. Health Check:")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        print(f"   Data loaded: {health['data_loaded']}")
    except Exception as e:
        print(f"   ❌ API not available: {e}")
        return
    
    # Get available teams
    print("\n2. Available Teams:")
    try:
        teams = client.get_teams()
        print(f"   Total teams: {len(teams)}")
        print(f"   First 10 teams: {teams[:10]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Get team statistics
    print("\n3. Team Statistics (Spain):")
    try:
        spain_stats = client.get_team_stats("Spain")
        print(f"   Team: {spain_stats['team_name']}")
        print(f"   Goals: {spain_stats['stats'].get('goals', 'N/A')}")
        print(f"   Attempts: {spain_stats['stats'].get('attempts', 'N/A')}")
        print(f"   Passes accuracy: {spain_stats['stats'].get('passes_accuracy', 'N/A')}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Single match prediction
    print("\n4. Single Match Prediction (Spain vs Italy):")
    try:
        prediction = client.predict_match("Spain", "Italy")
        print(f"   Match: {prediction['home_team']} vs {prediction['away_team']}")
        print(f"   Prediction: {prediction['predicted_result']}")
        print(f"   Expected goals: {prediction['predicted_goals']:.1f}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Probabilities:")
        for outcome, prob in prediction['probabilities'].items():
            print(f"     {outcome}: {prob:.1%}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Batch predictions
    print("\n5. Batch Predictions:")
    matches = [
        {"home_team": "England", "away_team": "France"},
        {"home_team": "Germany", "away_team": "Netherlands"},
        {"home_team": "Portugal", "away_team": "Spain"}
    ]
    
    try:
        batch_result = client.predict_batch(matches)
        print(f"   Predicted {len(batch_result['predictions'])} matches:")
        
        for pred in batch_result['predictions']:
            result_text = {
                '1': f"{pred['home_team']} WIN",
                '2': f"{pred['away_team']} WIN", 
                'X': "DRAW"
            }.get(pred['predicted_result'], pred['predicted_result'])
            
            print(f"     {pred['home_team']} vs {pred['away_team']}: {result_text} ({pred['confidence']:.1%})")
        
        print(f"\n   Summary:")
        summary = batch_result['summary']
        print(f"     Average goals: {summary['average_goals']}")
        print(f"     High confidence: {summary['high_confidence_predictions']}")
        print(f"     Over 2.5 goals: {summary['over_2_5_goals']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()