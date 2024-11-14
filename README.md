# NFL Game Outcome Prediction

## Overview
This project aims to build a machine learning model to predict NFL game outcomes, focusing on whether the home team wins or loses. The data includes game details, team information, betting spreads, and weather conditions. A Random Forest model is used to train on these features and predict game outcomes.

## Key Features
Team Indexing: Consistent team representation across team_home, team_away, and team_favorite_id columns using a manual indexing system.
Feature Engineering: Includes game-specific and weather-based features to enhance model performance.
Target Variable: The model predicts the team_home_result, where 1 indicates a home win and 0 indicates a loss.
