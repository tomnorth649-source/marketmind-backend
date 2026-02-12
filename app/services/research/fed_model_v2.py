"""Fed Model V2 - Meeting-level data with proper signal interpretation.

Key fixes:
1. Yield curve inversion is LEADING indicator (6-month lag)
2. Use TREND in indicators, not just levels
3. Claims trend > claims level
4. Combine with market expectations
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import httpx


class FedModelV2:
    """Improved Fed prediction model with meeting-level data."""
    
    FRED_BASE = "https://api.stlouisfed.org/fred"
    
    def __init__(self, fred_api_key: str):
        self.fred_api_key = fred_api_key
        self._cache = {}
    
    async def _get_series_history(
        self, 
        series_id: str, 
        end_date: str,
        periods: int = 12
    ) -> list[float]:
        """Get historical values for trend analysis."""
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=periods * 35)  # ~monthly data
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.FRED_BASE}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "observation_start": start.strftime("%Y-%m-%d"),
                    "observation_end": end_date,
                    "sort_order": "desc",
                    "limit": periods,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        
        values = []
        for obs in data.get("observations", []):
            if obs["value"] != ".":
                values.append(float(obs["value"]))
        
        return values
    
    def _calc_trend(self, values: list[float], periods: int = 3) -> str:
        """Calculate trend from recent values."""
        if len(values) < periods:
            return "neutral"
        
        recent = values[:periods]  # Most recent
        older = values[periods:periods*2] if len(values) >= periods*2 else values[periods:]
        
        if not older:
            return "neutral"
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        change_pct = (recent_avg - older_avg) / older_avg * 100 if older_avg else 0
        
        if change_pct > 5:
            return "rising"
        elif change_pct < -5:
            return "falling"
        return "stable"
    
    async def get_meeting_signals(self, meeting_date: str) -> dict:
        """Get all signals for a specific meeting date."""
        lookup = (datetime.strptime(meeting_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Fetch histories in parallel
        results = await asyncio.gather(
            self._get_series_history("T10Y2Y", lookup, 12),      # Yield curve
            self._get_series_history("ICSA", lookup, 8),         # Claims
            self._get_series_history("UNRATE", lookup, 6),       # Unemployment
            self._get_series_history("PCEPILFE", lookup, 13),    # Core PCE
            return_exceptions=True,
        )
        
        signals = {}
        
        # Yield curve - check if RECENTLY inverted (within 6 months)
        yc = results[0] if not isinstance(results[0], Exception) else []
        if yc:
            current_yc = yc[0]
            # Check last 6 months for inversion
            recent_inversions = sum(1 for v in yc[:6] if v < 0)
            
            if current_yc < 0:
                signals["yield_curve"] = {"value": current_yc, "signal": "inverted_now", "dovish_weight": 0.5}
            elif recent_inversions >= 3:
                signals["yield_curve"] = {"value": current_yc, "signal": "recently_inverted", "dovish_weight": 1.5}  # STRONG dovish
            else:
                signals["yield_curve"] = {"value": current_yc, "signal": "normal", "dovish_weight": 0}
        
        # Claims TREND - more important than level
        claims = results[1] if not isinstance(results[1], Exception) else []
        if claims:
            claims = [c/1000 for c in claims]  # Convert to thousands
            trend = self._calc_trend(claims)
            current = claims[0]
            
            if trend == "rising" and current > 230:
                signals["claims"] = {"value": current, "trend": trend, "signal": "deteriorating", "dovish_weight": 1.5}
            elif trend == "rising":
                signals["claims"] = {"value": current, "trend": trend, "signal": "weakening", "dovish_weight": 0.5}
            elif current > 260:
                signals["claims"] = {"value": current, "trend": trend, "signal": "elevated", "dovish_weight": 1.0}
            else:
                signals["claims"] = {"value": current, "trend": trend, "signal": "healthy", "dovish_weight": 0}
        
        # Unemployment TREND
        unemp = results[2] if not isinstance(results[2], Exception) else []
        if unemp:
            trend = self._calc_trend(unemp, 2)
            current = unemp[0]
            
            if trend == "rising":
                signals["unemployment"] = {"value": current, "trend": trend, "signal": "rising", "dovish_weight": 1.0}
            elif current > 4.5:
                signals["unemployment"] = {"value": current, "trend": trend, "signal": "elevated", "dovish_weight": 0.5}
            else:
                signals["unemployment"] = {"value": current, "trend": trend, "signal": "low", "dovish_weight": 0}
        
        # Inflation TREND (Fed cares about trajectory)
        pce = results[3] if not isinstance(results[3], Exception) else []
        if pce and len(pce) >= 13:
            # Calculate YoY
            current_yoy = ((pce[0] / pce[12]) - 1) * 100
            prev_yoy = ((pce[1] / pce[12]) - 1) * 100 if len(pce) > 12 else current_yoy
            
            if current_yoy < 2.5 and current_yoy < prev_yoy:
                signals["inflation"] = {"value": current_yoy, "trend": "falling", "signal": "at_target", "dovish_weight": 0.5}
            elif current_yoy > 3.0:
                signals["inflation"] = {"value": current_yoy, "trend": "elevated", "signal": "above_target", "hawkish_weight": 1.0}
            else:
                signals["inflation"] = {"value": current_yoy, "trend": "moderate", "signal": "near_target", "dovish_weight": 0}
        
        return signals
    
    def predict_from_signals(self, signals: dict) -> tuple[str, dict]:
        """Make prediction from signals."""
        probs = {
            "cut_50": 0.03,
            "cut_25": 0.15,
            "hold": 0.60,  # Higher base for hold
            "hike_25": 0.17,
            "hike_50": 0.05,
        }
        
        # Sum dovish/hawkish weights
        dovish_total = sum(
            s.get("dovish_weight", 0) 
            for s in signals.values()
        )
        hawkish_total = sum(
            s.get("hawkish_weight", 0) 
            for s in signals.values()
        )
        
        # Adjust probabilities
        if dovish_total >= 2.5:
            # Strong dovish signal
            probs["cut_50"] += 0.15
            probs["cut_25"] += 0.25
            probs["hold"] -= 0.25
            probs["hike_25"] -= 0.12
            probs["hike_50"] -= 0.03
        elif dovish_total >= 1.5:
            # Moderate dovish
            probs["cut_25"] += 0.20
            probs["hold"] -= 0.10
            probs["hike_25"] -= 0.08
            probs["hike_50"] -= 0.02
        elif dovish_total >= 0.5:
            # Slight dovish
            probs["cut_25"] += 0.10
            probs["hold"] -= 0.05
            probs["hike_25"] -= 0.05
        
        if hawkish_total >= 1.5:
            # Hawkish (inflation elevated)
            probs["hike_25"] += 0.15
            probs["hold"] -= 0.05
            probs["cut_25"] -= 0.08
            probs["cut_50"] -= 0.02
        
        # Normalize
        total = sum(probs.values())
        probs = {k: max(0.01, v/total) for k, v in probs.items()}
        
        predicted = max(probs, key=probs.get)
        return predicted, probs


async def run_v2_backtest(fred_api_key: str) -> dict:
    """Run V2 model backtest."""
    model = FedModelV2(fred_api_key)
    
    decisions = [
        ("2024-01-31", "hold"), ("2024-03-20", "hold"), ("2024-05-01", "hold"),
        ("2024-06-12", "hold"), ("2024-07-31", "hold"), ("2024-09-18", "cut_50"),
        ("2024-11-07", "cut_25"), ("2024-12-18", "cut_25"), ("2025-01-29", "hold"),
        ("2025-03-19", "hold"), ("2025-05-07", "cut_25"), ("2025-06-18", "hold"),
        ("2025-07-30", "hold"), ("2025-09-17", "cut_25"), ("2025-11-05", "cut_25"),
        ("2025-12-17", "hold"), ("2026-01-28", "hold"),
    ]
    
    results = []
    correct = 0
    
    for date, actual in decisions:
        signals = await model.get_meeting_signals(date)
        predicted, probs = model.predict_from_signals(signals)
        
        is_correct = predicted == actual
        if is_correct:
            correct += 1
        
        dovish = sum(s.get("dovish_weight", 0) for s in signals.values())
        
        results.append({
            "date": date,
            "predicted": predicted,
            "prob": f"{probs[predicted]*100:.0f}%",
            "actual": actual,
            "correct": is_correct,
            "dovish_score": round(dovish, 1),
        })
        
        await asyncio.sleep(0.3)  # Rate limit
    
    return {
        "accuracy": f"{correct/len(decisions)*100:.1f}%",
        "correct": f"{correct}/{len(decisions)}",
        "details": results,
    }
