"""Calibration Tracking API endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

from app.services.calibration import (
    get_calibration_tracker, 
    CalibrationTracker,
    Prediction,
    PredictionCategory,
)

router = APIRouter(prefix="/calibration", tags=["calibration"])


def get_tracker() -> CalibrationTracker:
    return get_calibration_tracker()


class PredictionCreate(BaseModel):
    """Schema for creating a new prediction."""
    category: str  # fed_rate, weather, politics, sports, crypto, other
    event_description: str
    resolution_date: str  # YYYY-MM-DD
    outcomes: dict[str, float]  # outcome_name -> probability (must sum to 1)
    market_prob: Optional[float] = None
    market_source: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "fed_rate",
                "event_description": "FOMC March 2026 - Rate Decision",
                "resolution_date": "2026-03-18",
                "outcomes": {
                    "cut_25": 0.15,
                    "hold": 0.78,
                    "hike_25": 0.07,
                },
                "market_prob": 0.78,
                "market_source": "kalshi",
            }
        }


class PredictionResolve(BaseModel):
    """Schema for resolving a prediction."""
    actual_outcome: str


@router.post("/predictions")
async def create_prediction(
    data: PredictionCreate,
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """
    Record a new prediction for calibration tracking.
    
    The outcomes dict should have probabilities that sum to 1.
    """
    # Validate outcomes sum to ~1
    total = sum(data.outcomes.values())
    if not (0.99 <= total <= 1.01):
        raise HTTPException(
            status_code=400,
            detail=f"Outcome probabilities must sum to 1 (got {total})"
        )
    
    # Find predicted outcome (highest probability)
    predicted_outcome = max(data.outcomes, key=data.outcomes.get)
    predicted_prob = data.outcomes[predicted_outcome]
    
    # Create prediction
    prediction = Prediction(
        id=str(uuid.uuid4())[:8],
        category=PredictionCategory(data.category),
        event_description=data.event_description,
        prediction_date=datetime.now().strftime("%Y-%m-%d"),
        resolution_date=data.resolution_date,
        outcomes=data.outcomes,
        predicted_outcome=predicted_outcome,
        predicted_prob=predicted_prob,
        market_prob=data.market_prob,
        market_source=data.market_source,
    )
    
    tracker.record_prediction(prediction)
    
    return {
        "id": prediction.id,
        "message": "Prediction recorded",
        "predicted_outcome": predicted_outcome,
        "predicted_prob": f"{predicted_prob*100:.1f}%",
        "resolution_date": data.resolution_date,
    }


@router.post("/predictions/{prediction_id}/resolve")
async def resolve_prediction(
    prediction_id: str,
    data: PredictionResolve,
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """
    Resolve a prediction with the actual outcome.
    
    This calculates accuracy and Brier score.
    """
    try:
        prediction = tracker.resolve_prediction(prediction_id, data.actual_outcome)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return {
        "id": prediction.id,
        "event": prediction.event_description,
        "predicted": prediction.predicted_outcome,
        "predicted_prob": f"{prediction.predicted_prob*100:.1f}%",
        "actual": prediction.actual_outcome,
        "was_correct": prediction.was_correct,
        "brier_score": round(prediction.brier_score, 4),
        "brier_interpretation": _interpret_brier(prediction.brier_score),
    }


@router.get("/predictions")
async def list_predictions(
    category: Optional[str] = None,
    pending_only: bool = False,
    limit: int = 50,
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """List predictions, optionally filtered by category."""
    if pending_only:
        predictions = tracker.get_pending_predictions(category)
    else:
        # Get all predictions (need to implement this method)
        predictions = tracker.get_pending_predictions(category)  # Placeholder
    
    return {
        "count": len(predictions),
        "predictions": [p.to_dict() for p in predictions[:limit]],
    }


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """Get a single prediction by ID."""
    prediction = tracker.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction.to_dict()


@router.get("/stats")
async def get_stats(
    category: Optional[str] = None,
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """
    Get calibration statistics.
    
    Returns:
        - Total/resolved/correct predictions
        - Accuracy percentage
        - Average Brier score
        - Calibration by probability bucket
        - Overconfidence score
    """
    stats = tracker.get_stats(category)
    
    return {
        "category": stats.category or "all",
        "total_predictions": stats.total_predictions,
        "resolved_predictions": stats.resolved_predictions,
        "correct_predictions": stats.correct_predictions,
        "accuracy": f"{stats.accuracy*100:.1f}%",
        "avg_brier_score": stats.avg_brier_score,
        "brier_interpretation": _interpret_brier(stats.avg_brier_score),
        "calibration_by_bucket": stats.calibration_by_bucket,
        "overconfidence_score": stats.overconfidence_score,
        "overconfidence_interpretation": _interpret_overconfidence(stats.overconfidence_score),
    }


@router.get("/health")
async def calibration_health(
    tracker: CalibrationTracker = Depends(get_tracker),
):
    """
    Quick health check for calibration system.
    
    Returns summary of how well-calibrated we are.
    """
    stats = tracker.get_stats()
    
    # Determine overall health
    if stats.resolved_predictions < 10:
        health = "insufficient_data"
        message = "Need more resolved predictions for meaningful stats"
    elif abs(stats.overconfidence_score) < 0.05 and stats.avg_brier_score < 0.2:
        health = "excellent"
        message = "Well-calibrated predictions"
    elif abs(stats.overconfidence_score) < 0.10 and stats.avg_brier_score < 0.3:
        health = "good"
        message = "Reasonably calibrated"
    elif stats.overconfidence_score > 0.15:
        health = "overconfident"
        message = "Predictions are too confident - consider widening uncertainty"
    elif stats.overconfidence_score < -0.15:
        health = "underconfident"
        message = "Predictions are too uncertain - can be more confident"
    else:
        health = "moderate"
        message = "Room for improvement in calibration"
    
    return {
        "health": health,
        "message": message,
        "total_predictions": stats.total_predictions,
        "resolved": stats.resolved_predictions,
        "accuracy": f"{stats.accuracy*100:.1f}%",
        "brier_score": stats.avg_brier_score,
        "overconfidence": stats.overconfidence_score,
    }


def _interpret_brier(score: float) -> str:
    """Interpret Brier score."""
    if score < 0.1:
        return "Excellent calibration"
    elif score < 0.2:
        return "Good calibration"
    elif score < 0.3:
        return "Moderate calibration"
    else:
        return "Poor calibration - review methodology"


def _interpret_overconfidence(score: float) -> str:
    """Interpret overconfidence score."""
    if abs(score) < 0.03:
        return "Well-calibrated"
    elif score > 0.10:
        return "Significantly overconfident"
    elif score > 0.05:
        return "Slightly overconfident"
    elif score < -0.10:
        return "Significantly underconfident"
    elif score < -0.05:
        return "Slightly underconfident"
    else:
        return "Mildly biased"
