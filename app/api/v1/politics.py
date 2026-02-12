"""Politics Research API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.research.politics import get_politics_module

router = APIRouter(prefix="/politics", tags=["politics-research"])


@router.get("/dashboard")
async def get_politics_dashboard():
    """Get political research dashboard."""
    module = get_politics_module()
    
    try:
        return await module.get_politics_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/confirmation")
async def estimate_confirmation(
    nominee: str = Query(..., description="Nominee name"),
    position: str = Query(..., description="Position (e.g., 'Fed Chair', 'Supreme Court')"),
    senate_margin: int = Query(default=2, description="Senate margin (+R/-D)"),
    controversy: str = Query(default="low", description="Controversy level: low/medium/high"),
):
    """
    Estimate probability of Senate confirmation.
    
    Used for Kalshi markets like:
    - "Will Kevin Warsh be confirmed as Fed Chair?"
    """
    if controversy not in ["low", "medium", "high"]:
        raise HTTPException(status_code=400, detail="controversy must be low/medium/high")
    
    module = get_politics_module()
    
    try:
        return module.estimate_confirmation_probability(
            nominee, position, senate_margin, controversy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/policy")
async def estimate_policy(
    policy: str = Query(..., description="Policy/legislation name"),
    status: str = Query(default="proposed", description="Current status"),
    bipartisan: str = Query(default="low", description="Bipartisan support: low/medium/high"),
):
    """
    Estimate probability of policy/legislation passing.
    
    Status options: proposed, committee, passed_house, passed_senate, conference
    """
    module = get_politics_module()
    
    try:
        return module.estimate_policy_probability(policy, status, bipartisan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/approval")
async def estimate_approval(
    current: float = Query(..., ge=0, le=100, description="Current approval rating"),
    trend: float = Query(default=0, description="Monthly trend (+/- points)"),
):
    """
    Estimate probability of approval rating above/below thresholds.
    
    Returns probabilities for 40%, 45%, 50%, 55% thresholds.
    """
    module = get_politics_module()
    
    try:
        probs = module.estimate_approval_rating(current, trend)
        return {
            "current_approval": current,
            "trend": f"{trend:+.1f} pts/month",
            "probabilities": probs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
