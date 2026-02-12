"""Calibration Tracking System.

Tracks prediction accuracy over time to:
1. Measure how well our probabilities match reality
2. Identify systematic biases (over/under-confidence)
3. Build trust through transparency

A well-calibrated system means:
- When we say 70% probability, it happens ~70% of the time
- Brier scores close to 0 are better
"""
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from typing import Optional
import json
from pathlib import Path
import sqlite3


class PredictionCategory(str, Enum):
    FED_RATE = "fed_rate"
    WEATHER = "weather"
    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    OTHER = "other"


@dataclass
class Prediction:
    """A single prediction to track."""
    id: str
    category: PredictionCategory
    event_description: str
    prediction_date: str  # When we made the prediction
    resolution_date: str  # When we'll know the outcome
    
    # Our probability estimates
    outcomes: dict[str, float]  # outcome_name -> probability
    predicted_outcome: str  # Most likely outcome
    predicted_prob: float  # Probability we assigned
    
    # Market comparison (optional)
    market_prob: Optional[float] = None  # What the market said
    market_source: Optional[str] = None  # "kalshi", "polymarket", etc.
    
    # Resolution (filled in later)
    actual_outcome: Optional[str] = None
    resolved_at: Optional[str] = None
    was_correct: Optional[bool] = None
    brier_score: Optional[float] = None  # Lower is better (0 = perfect)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        d["category"] = PredictionCategory(d["category"])
        return cls(**d)


@dataclass
class CalibrationStats:
    """Calibration statistics for a category or overall."""
    category: Optional[str]
    total_predictions: int
    resolved_predictions: int
    correct_predictions: int
    accuracy: float  # % correct
    avg_brier_score: float  # Lower is better
    calibration_by_bucket: dict[str, dict]  # "70-80%" -> {predicted: 0.75, actual: 0.72}
    overconfidence_score: float  # >0 means overconfident, <0 means underconfident
    edge_vs_market: Optional[float]  # Our accuracy - market accuracy (when available)


class CalibrationTracker:
    """Track and analyze prediction calibration."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "calibration.db"
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                event_description TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                resolution_date TEXT NOT NULL,
                outcomes_json TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                predicted_prob REAL NOT NULL,
                market_prob REAL,
                market_source TEXT,
                actual_outcome TEXT,
                resolved_at TEXT,
                was_correct INTEGER,
                brier_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def record_prediction(self, prediction: Prediction) -> str:
        """Record a new prediction."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO predictions (
                id, category, event_description, prediction_date, resolution_date,
                outcomes_json, predicted_outcome, predicted_prob, market_prob, market_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.id,
            prediction.category.value,
            prediction.event_description,
            prediction.prediction_date,
            prediction.resolution_date,
            json.dumps(prediction.outcomes),
            prediction.predicted_outcome,
            prediction.predicted_prob,
            prediction.market_prob,
            prediction.market_source,
        ))
        conn.commit()
        conn.close()
        return prediction.id
    
    def resolve_prediction(self, prediction_id: str, actual_outcome: str) -> Prediction:
        """
        Resolve a prediction with the actual outcome.
        
        Calculates:
        - was_correct: Did we predict the right outcome?
        - brier_score: How well calibrated was our probability?
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", 
            (prediction_id,)
        ).fetchone()
        
        if not row:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        # Parse the prediction
        outcomes = json.loads(row[5])
        predicted_outcome = row[6]
        predicted_prob = row[7]
        
        # Calculate metrics
        was_correct = actual_outcome == predicted_outcome
        
        # Brier score: (forecast - outcome)^2
        # outcome = 1 if actual matches predicted, else 0
        outcome_binary = 1.0 if was_correct else 0.0
        brier_score = (predicted_prob - outcome_binary) ** 2
        
        # Update database
        conn.execute("""
            UPDATE predictions 
            SET actual_outcome = ?, resolved_at = ?, was_correct = ?, brier_score = ?
            WHERE id = ?
        """, (
            actual_outcome,
            datetime.now().isoformat(),
            1 if was_correct else 0,
            brier_score,
            prediction_id,
        ))
        conn.commit()
        conn.close()
        
        return self.get_prediction(prediction_id)
    
    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Get a single prediction by ID."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,)
        ).fetchone()
        conn.close()
        
        if not row:
            return None
        
        return Prediction(
            id=row[0],
            category=PredictionCategory(row[1]),
            event_description=row[2],
            prediction_date=row[3],
            resolution_date=row[4],
            outcomes=json.loads(row[5]),
            predicted_outcome=row[6],
            predicted_prob=row[7],
            market_prob=row[8],
            market_source=row[9],
            actual_outcome=row[10],
            resolved_at=row[11],
            was_correct=bool(row[12]) if row[12] is not None else None,
            brier_score=row[13],
        )
    
    def get_pending_predictions(self, category: str = None) -> list[Prediction]:
        """Get predictions that need resolution."""
        conn = sqlite3.connect(self.db_path)
        
        if category:
            rows = conn.execute("""
                SELECT * FROM predictions 
                WHERE actual_outcome IS NULL AND category = ?
                ORDER BY resolution_date ASC
            """, (category,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM predictions 
                WHERE actual_outcome IS NULL
                ORDER BY resolution_date ASC
            """).fetchall()
        
        conn.close()
        return [self._row_to_prediction(r) for r in rows]
    
    def get_stats(self, category: str = None) -> CalibrationStats:
        """Get calibration statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Base query
        where = "WHERE 1=1"
        params = []
        if category:
            where += " AND category = ?"
            params.append(category)
        
        # Total predictions
        total = conn.execute(
            f"SELECT COUNT(*) FROM predictions {where}", params
        ).fetchone()[0]
        
        # Resolved predictions
        resolved = conn.execute(
            f"SELECT COUNT(*) FROM predictions {where} AND actual_outcome IS NOT NULL",
            params
        ).fetchone()[0]
        
        # Correct predictions
        correct = conn.execute(
            f"SELECT COUNT(*) FROM predictions {where} AND was_correct = 1",
            params
        ).fetchone()[0]
        
        # Average Brier score
        avg_brier = conn.execute(
            f"SELECT AVG(brier_score) FROM predictions {where} AND brier_score IS NOT NULL",
            params
        ).fetchone()[0] or 0.0
        
        # Calibration by probability bucket
        calibration_buckets = {}
        buckets = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        bucket_names = ["0-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
        
        for (low, high), name in zip(buckets, bucket_names):
            bucket_rows = conn.execute(f"""
                SELECT predicted_prob, was_correct FROM predictions 
                {where} AND predicted_prob >= ? AND predicted_prob < ?
                AND was_correct IS NOT NULL
            """, params + [low, high]).fetchall()
            
            if bucket_rows:
                avg_predicted = sum(r[0] for r in bucket_rows) / len(bucket_rows)
                actual_rate = sum(r[1] for r in bucket_rows) / len(bucket_rows)
                calibration_buckets[name] = {
                    "count": len(bucket_rows),
                    "avg_predicted": round(avg_predicted, 3),
                    "actual_rate": round(actual_rate, 3),
                    "gap": round(avg_predicted - actual_rate, 3),
                }
        
        conn.close()
        
        # Calculate overconfidence score
        # Positive = overconfident (predicted higher than actual)
        gaps = [b["gap"] for b in calibration_buckets.values() if "gap" in b]
        overconfidence = sum(gaps) / len(gaps) if gaps else 0.0
        
        accuracy = correct / resolved if resolved > 0 else 0.0
        
        return CalibrationStats(
            category=category,
            total_predictions=total,
            resolved_predictions=resolved,
            correct_predictions=correct,
            accuracy=round(accuracy, 3),
            avg_brier_score=round(avg_brier, 4),
            calibration_by_bucket=calibration_buckets,
            overconfidence_score=round(overconfidence, 3),
            edge_vs_market=None,  # TODO: Calculate when we have market data
        )
    
    def _row_to_prediction(self, row) -> Prediction:
        """Convert database row to Prediction."""
        return Prediction(
            id=row[0],
            category=PredictionCategory(row[1]),
            event_description=row[2],
            prediction_date=row[3],
            resolution_date=row[4],
            outcomes=json.loads(row[5]),
            predicted_outcome=row[6],
            predicted_prob=row[7],
            market_prob=row[8],
            market_source=row[9],
            actual_outcome=row[10],
            resolved_at=row[11],
            was_correct=bool(row[12]) if row[12] is not None else None,
            brier_score=row[13],
        )


# Singleton
_tracker: CalibrationTracker | None = None

def get_calibration_tracker(db_path: str = None) -> CalibrationTracker:
    global _tracker
    if _tracker is None:
        _tracker = CalibrationTracker(db_path)
    return _tracker
