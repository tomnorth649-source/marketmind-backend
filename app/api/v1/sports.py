"""Sports Research API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.research.sports import get_sports_module

router = APIRouter(prefix="/sports", tags=["sports-research"])


@router.get("/dashboard/{league}")
async def get_sports_dashboard(league: str):
    """
    Get sports dashboard for a league.
    
    Available leagues: nba, nfl, mlb, nhl, ncaab, ncaaf, mls
    """
    module = get_sports_module()
    
    try:
        return await module.get_sports_dashboard(league.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/games/{league}")
async def get_games(
    league: str,
    date: Optional[str] = Query(default=None, description="Date in YYYYMMDD format"),
):
    """Get games for a league on a specific date."""
    module = get_sports_module()
    
    try:
        games = await module.get_games(league.lower(), date)
        return {
            "league": league.upper(),
            "date": date or "today",
            "count": len(games),
            "games": [
                {
                    "id": g.id,
                    "home_team": g.home_team.name,
                    "away_team": g.away_team.name,
                    "start_time": g.start_time.isoformat(),
                    "status": g.status,
                    "home_score": g.home_score,
                    "away_score": g.away_score,
                    "venue": g.venue,
                }
                for g in games
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/standings/{league}")
async def get_standings(league: str):
    """Get current standings for a league."""
    module = get_sports_module()
    
    try:
        return await module.get_standings(league.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/game")
async def get_game_probability(
    league: str = Query(..., description="League code (nba, nfl, etc.)"),
    home: str = Query(..., description="Home team name or abbreviation"),
    away: str = Query(..., description="Away team name or abbreviation"),
):
    """
    Estimate win probability for a game.
    
    Uses home advantage baseline, team records, and standings.
    """
    module = get_sports_module()
    
    try:
        return await module.estimate_game_probability(league.lower(), home, away)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leagues")
async def list_leagues():
    """List available leagues."""
    module = get_sports_module()
    return {
        "leagues": [
            {"code": code, "name": info["name"]}
            for code, info in module.LEAGUES.items()
        ]
    }
