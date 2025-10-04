#!/usr/bin/env python3
"""
Euro 2024 Match Prediction Script
=================================

This script uses the trained models to predict Euro 2024 match results and goals.
Compatible with both original and enhanced model versions.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class Euro2024Predictor:
    def __init__(self, data_path='../data/', model_version='v2_enhanced'):
        self.data_path = data_path
        self.model_version = model_version
        self.result_model = None
        self.goals_model = None
        self.feature_names = None
        
    def load_models(self):
        """Load trained models"""
        try:
            # Try to load enhanced models first
            result_filename = f"{self.data_path}euro24_results_{self.model_version}.joblib"
            goals_filename = f"{self.data_path}euro24_goals_{self.model_version}.joblib"
            feature_filename = f"{self.data_path}feature_names_{self.model_version}.joblib"
            
            self.result_model = joblib.load(result_filename)
            self.goals_model = joblib.load(goals_filename)
            self.feature_names = joblib.load(feature_filename)
            
            print(f"✅ Loaded enhanced models ({self.model_version})")
            
        except FileNotFoundError:
            # Fallback to original models
            try:
                result_filename = f"{self.data_path}euro20_results.joblib"
                goals_filename = f"{self.data_path}euro20_goals.joblib"
                
                self.result_model = joblib.load(result_filename)
                self.goals_model = joblib.load(goals_filename)
                
                print("✅ Loaded original models (euro20)")
                print("⚠️  Note: Using original feature set")
                
            except FileNotFoundError:
                raise FileNotFoundError("No trained models found! Please run training first.")
    
    def load_euro24_data(self):
        """Load Euro 2024 team statistics and match fixtures"""
        # Load team stats
        df_qual_24 = pd.read_csv(f'{self.data_path}euro24_qualifiers_norm.csv', index_col=0)
        
        # Create synthetic Euro 2024 matches for demonstration
        # In a real scenario, you would load actual fixture data
        teams_24 = [
            'Germany', 'Scotland', 'Hungary', 'Switzerland',
            'Spain', 'Croatia', 'Italy', 'Albania', 
            'Slovenia', 'Denmark', 'Serbia', 'England',
            'Poland', 'Netherlands', 'Austria', 'France',
            'Belgium', 'Slovakia', 'Romania', 'Ukraine',
            'Türkiye', 'Georgia', 'Portugal', 'Czechia'
        ]
        
        # Create group stage matches (simplified)
        matches = []
        group_matches = [
            ('Germany', 'Scotland'), ('Hungary', 'Switzerland'),
            ('Spain', 'Croatia'), ('Italy', 'Albania'),
            ('Slovenia', 'Denmark'), ('Serbia', 'England'),
            ('Poland', 'Netherlands'), ('Austria', 'France'),
            ('Belgium', 'Slovakia'), ('Romania', 'Ukraine'),
            ('Türkiye', 'Georgia'), ('Portugal', 'Czechia')
        ]
        
        for i, (home, away) in enumerate(group_matches):
            match = {
                'home_team': home,
                'away_team': away,
                'date': f'2024-06-{15 + i % 10}',
                'city': f'City_{i+1}',
                'tournament': 'UEFA Euro'
            }
            matches.append(match)
        
        df_group_24 = pd.DataFrame(matches)
        
        # Normalize team names
        df_group_24.replace('Czech Republic', 'Czechia', inplace=True)
        df_group_24.replace('Turkey', 'Türkiye', inplace=True)
        
        return df_group_24, df_qual_24
    
    def create_match_features(self, df_group_24, df_qual_24):
        """Create feature matrix for Euro 2024 matches"""
        all_stats = []
        match_info = []
        
        for idx, row in df_group_24.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            home_stats = df_qual_24[df_qual_24['team_name'] == home_team].copy()
            away_stats = df_qual_24[df_qual_24['team_name'] == away_team].copy()
            
            if len(home_stats) == 0 or len(away_stats) == 0:
                print(f"⚠️  Skipping {home_team} vs {away_team} - missing data")
                continue
            
            match_stats = []
            
            # Remove team_name column before adding suffix (same as training)
            home_stats = home_stats.drop('team_name', axis=1)
            away_stats = away_stats.drop('team_name', axis=1)
            
            home_stats = home_stats.add_suffix('_home')
            away_stats = away_stats.add_suffix('_away')
            
            match_stats.extend(home_stats.values[0])
            match_stats.extend(away_stats.values[0])
            
            # Add enhanced features if using enhanced model
            if self.feature_names and 'home_attack_strength' in self.feature_names:
                home_attack_strength = home_stats['goals_home'].values[0] / home_stats['attempts_home'].values[0] if home_stats['attempts_home'].values[0] > 0 else 0
                away_attack_strength = away_stats['goals_away'].values[0] / away_stats['attempts_away'].values[0] if away_stats['attempts_away'].values[0] > 0 else 0
                home_defense_strength = 1 - (home_stats['goals_conceded_home'].values[0] / 10)
                away_defense_strength = 1 - (away_stats['goals_conceded_away'].values[0] / 10)
                
                match_stats.extend([
                    home_attack_strength, away_attack_strength,
                    home_defense_strength, away_defense_strength,
                    home_attack_strength - away_defense_strength,
                    away_attack_strength - home_defense_strength
                ])
            
            all_stats.append(match_stats)
            match_info.append({
                'home_team': home_team,
                'away_team': away_team,
                'date': row['date'],
                'city': row['city']
            })
        
        # Create DataFrame with exact feature names from training
        if self.feature_names:
            # Use the exact feature names from the trained model
            df_features = pd.DataFrame(all_stats, columns=self.feature_names)
        else:
            # Fallback for original model
            home_cols = [col for col in home_stats.columns]
            away_cols = [col for col in away_stats.columns]
            columns = home_cols + away_cols
            df_features = pd.DataFrame(all_stats, columns=columns)
        
        df_match_info = pd.DataFrame(match_info)
        
        return df_features, df_match_info
    
    def predict_matches(self, X, match_info):
        """Make predictions for all matches"""
        # Predict results
        result_predictions = self.result_model.predict(X)
        result_probabilities = self.result_model.predict_proba(X)
        
        # Predict goals
        goal_predictions = self.goals_model.predict(X)
        
        # Decode result predictions
        result_labels = ['X' if pred == 0 else str(pred) for pred in result_predictions]
        
        # Create results DataFrame
        results = match_info.copy()
        results['predicted_result'] = result_labels
        results['predicted_goals'] = goal_predictions
        
        # Add probabilities
        results['prob_draw'] = result_probabilities[:, 0]
        results['prob_home_win'] = result_probabilities[:, 1] if result_probabilities.shape[1] > 1 else 0
        results['prob_away_win'] = result_probabilities[:, 2] if result_probabilities.shape[1] > 2 else 0
        
        # Calculate confidence (max probability)
        results['confidence'] = np.max(result_probabilities, axis=1)
        
        return results
    
    def display_predictions(self, predictions):
        """Display predictions in a formatted way"""
        print("\n" + "="*80)
        print("🏆 EURO 2024 MATCH PREDICTIONS")
        print("="*80)
        
        # Group by date
        for date in predictions['date'].unique():
            date_matches = predictions[predictions['date'] == date]
            print(f"\n📅 {date}")
            print("-" * 60)
            
            for _, match in date_matches.iterrows():
                home = match['home_team']
                away = match['away_team']
                result = match['predicted_result']
                goals = match['predicted_goals']
                confidence = match['confidence']
                city = match['city']
                
                # Determine prediction explanation
                if result == '1':
                    prediction_text = f"{home} WIN"
                elif result == '2':
                    prediction_text = f"{away} WIN"
                else:
                    prediction_text = "DRAW"
                
                print(f"🏟️  {home} vs {away} ({city})")
                print(f"   Prediction: {prediction_text} | Goals: {goals:.1f} | Confidence: {confidence:.1%}")
                
                # Show probabilities
                print(f"   Probabilities - Home: {match['prob_home_win']:.1%} | " +
                     f"Draw: {match['prob_draw']:.1%} | Away: {match['prob_away_win']:.1%}")
                print()
        
        # Summary statistics
        high_confidence = predictions[predictions['confidence'] >= 0.6]
        over_2_5 = predictions[predictions['predicted_goals'] > 2.5]
        
        print("\n" + "="*80)
        print("📊 PREDICTION SUMMARY")
        print("="*80)
        print(f"Total matches predicted: {len(predictions)}")
        print(f"High confidence predictions (≥60%): {len(high_confidence)}")
        print(f"Over 2.5 goals predicted: {len(over_2_5)}")
        print(f"Average goals per match: {predictions['predicted_goals'].mean():.2f}")
        
        # Best bets (high confidence)
        if len(high_confidence) > 0:
            print(f"\n🎯 HIGH CONFIDENCE PREDICTIONS:")
            print("-" * 40)
            for _, match in high_confidence.iterrows():
                home = match['home_team']
                away = match['away_team']
                result = match['predicted_result']
                confidence = match['confidence']
                
                if result == '1':
                    bet_text = f"{home} to win"
                elif result == '2':
                    bet_text = f"{away} to win"
                else:
                    bet_text = "Draw"
                
                print(f"   {home} vs {away}: {bet_text} ({confidence:.1%})")
    
    def save_predictions(self, predictions, filename=None):
        """Save predictions to CSV"""
        if filename is None:
            filename = f"{self.data_path}euro2024_predictions_{self.model_version}.csv"
        
        predictions.to_csv(filename, index=False)
        print(f"\n💾 Predictions saved to: {filename}")
    
    def run_prediction(self, save_results=True):
        """Run complete prediction pipeline"""
        print("🚀 Starting Euro 2024 Match Predictions")
        print("="*50)
        
        # Load models
        self.load_models()
        
        # Load data
        df_group_24, df_qual_24 = self.load_euro24_data()
        print(f"📊 Loaded {len(df_group_24)} Euro 2024 matches")
        
        # Create features
        X, match_info = self.create_match_features(df_group_24, df_qual_24)
        print(f"🔧 Created features for {len(X)} matches")
        
        # Make predictions
        predictions = self.predict_matches(X, match_info)
        
        # Display results
        self.display_predictions(predictions)
        
        # Save results
        if save_results:
            self.save_predictions(predictions)
        
        return predictions

if __name__ == "__main__":
    # Run predictions
    predictor = Euro2024Predictor()
    results = predictor.run_prediction()