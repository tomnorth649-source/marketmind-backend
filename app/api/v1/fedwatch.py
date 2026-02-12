"""FedWatch API endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import os

from app.services.research.fedwatch import get_fedwatch_client, CMEFedWatchClient

router = APIRouter(prefix="/fedwatch", tags=["fedwatch"])


def get_client() -> CMEFedWatchClient:
    """Get FedWatch client with API key if available."""
    api_key = os.getenv("CME_FEDWATCH_API_KEY")
    return get_fedwatch_client(api_key)


@router.get("/probabilities")
async def get_probabilities(
    meeting_date: Optional[str] = None,
    client: CMEFedWatchClient = Depends(get_client),
):
    """
    Get FedWatch probabilities for a specific FOMC meeting.
    
    Args:
        meeting_date: FOMC meeting date (YYYY-MM-DD). Defaults to next meeting.
    
    Returns:
        Probability distribution for rate outcomes.
    """
    try:
        probs = await client.get_probabilities(meeting_date)
        return probs.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline")
async def get_timeline(
    client: CMEFedWatchClient = Depends(get_client),
):
    """
    Get FedWatch probabilities for all upcoming FOMC meetings.
    
    Returns:
        - All meeting probabilities
        - Current rate
        - Rate path (implied rates by meeting)
        - Total cuts priced for year
    """
    try:
        timeline = await client.get_timeline()
        return {
            "meetings": [m.to_dict() for m in timeline.meetings],
            "current_rate": timeline.current_rate,
            "rate_path": timeline.rate_path,
            "cuts_priced_ytd": timeline.cuts_priced_ytd,
            "updated_at": timeline.updated_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next-meeting")
async def get_next_meeting(
    client: CMEFedWatchClient = Depends(get_client),
):
    """
    Get FedWatch data for the next FOMC meeting only.
    
    Simplified endpoint for quick checks.
    """
    try:
        probs = await client.get_probabilities()
        
        # Interpret the data for easy consumption
        implied_change = probs.implied_rate - probs.current_target_high
        
        if implied_change < -0.20:
            outlook = "Markets pricing significant cuts"
        elif implied_change < -0.05:
            outlook = "Markets pricing modest cut probability"
        elif implied_change > 0.05:
            outlook = "Markets pricing hike probability"
        else:
            outlook = "Markets pricing hold"
        
        return {
            "meeting_date": probs.meeting_date,
            "current_rate": f"{probs.current_target_low:.2f}-{probs.current_target_high:.2f}%",
            "most_likely": {
                "outcome": probs.most_likely_range,
                "probability": f"{probs.most_likely_prob*100:.1f}%",
            },
            "implied_rate": f"{probs.implied_rate:.3f}%",
            "implied_change_bps": round(implied_change * 100, 1),
            "outlook": outlook,
            "source": probs.source,
            "updated_at": probs.updated_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
