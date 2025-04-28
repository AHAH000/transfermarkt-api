import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# Initialize FastAPI Router (NOT FastAPI app)
router = APIRouter()

# Load dataset once (to avoid reloading on every request)
file_path = "Final_Player_Data_With_Market_Value.csv"
df_uploaded = pd.read_csv(file_path)

# Ensure 'contract_years_remaining' is derived from 'contract_expiration_date'
current_year = datetime.now().year
df_uploaded["contract_expiration_year"] = pd.to_numeric(df_uploaded["contract_expiration_date"].str[-4:], errors='coerce')
df_uploaded["contract_years_remaining"] = df_uploaded["contract_expiration_year"] - current_year
df_uploaded["contract_years_remaining"] = df_uploaded["contract_years_remaining"].fillna(0).astype(int)

# Define request model
class PlayerInput(BaseModel):
    age: int
    position: str
    sub_position: str = None  # Optional
    foot_encoded: int
    height_in_cm: int
    league: str
    contract_years_remaining: int

@router.post("/predict/")
def find_most_similar_players(target_player: PlayerInput, top_n=3):
    """Find the top N most similar players and estimate market value."""

    # Filter dataset by position
    df_filtered = df_uploaded[df_uploaded["position"] == target_player.position].copy()

    # Apply sub_position filter if provided
    if target_player.sub_position:
        df_filtered = df_filtered[df_filtered["sub_position"] == target_player.sub_position].copy()

    # Further filter by league
    df_filtered = df_filtered[df_filtered["current_club_domestic_competition_id"] == target_player.league].copy()

    if df_filtered.empty:
        return {"Error": "No similar players found in the same league and position."}

    # Store original age and height before normalization
    original_ages = df_filtered["age"].copy()
    original_heights = df_filtered["height_in_cm"].copy()

    # Select relevant features
    search_features = ["age", "foot_encoded", "height_in_cm", "contract_years_remaining"]

    # Normalize features
    scaler = MinMaxScaler()
    df_filtered[search_features] = scaler.fit_transform(df_filtered[search_features])

    # Train Nearest Neighbors Model
    neighbor_model = NearestNeighbors(n_neighbors=top_n, metric="euclidean")
    neighbor_model.fit(df_filtered[search_features])

    # Prepare input data
    target_data = np.array([
        target_player.age,
        target_player.foot_encoded,
        target_player.height_in_cm,
        target_player.contract_years_remaining
    ]).reshape(1, -1)

    # Normalize target input
    target_data = scaler.transform(target_data)

    # Find nearest players
    distances, indices = neighbor_model.kneighbors(target_data)
    results = []
    market_values = []

    for i, idx in enumerate(indices[0]):
        most_similar = df_filtered.iloc[idx]
        similarity_score = max(0, 100 - (distances[0][i] * 10))  # Convert distance to percentage

        # Retrieve unnormalized age and height
        original_age = int(original_ages.iloc[idx])
        original_height = int(original_heights.iloc[idx])
        market_values.append(most_similar["highest_market_value_in_eur"])

        results.append({
            "Player ID": int(most_similar["player_id"]),
            "Most Similar Player": f"{most_similar['first_name']} {most_similar['last_name']}",
            "Club": most_similar["current_club_name"],
            "Market Value (€)": int(most_similar["highest_market_value_in_eur"]),  # Convert to int
            "Age": original_age,  # Unnormalized Age
            "Position": most_similar["position"],
            "Sub-Position": most_similar.get("sub_position", "N/A"),  
            "League": most_similar["current_club_domestic_competition_id"],
            "Foot": most_similar["foot"],
            "Height (cm)": original_height,  # Unnormalized Height
            "Contract Years Remaining": int(most_similar["contract_years_remaining"]),
            "Similarity Score (%)": round(similarity_score, 2)
        })

    # Calculate the average market value of similar players and round to the nearest million
    avg_market_value = np.mean(market_values) if market_values else 0
    rounded_market_value = round(avg_market_value, -6)  # Round to nearest million

    return {
        "Predicted Market Value (€)": int(rounded_market_value),
        "Similar Players": results
    }
