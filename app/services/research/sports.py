"""Sports Research Module.

Provides probability estimates for:
- Game outcomes (NBA, NFL, MLB, NHL)
- Championship/playoff predictions
- Player props (if data available)

Data sources:
- ESPN API (unofficial, free)
- The Odds API (for bookmaker lines)
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import httpx


@dataclass
class Team:
    """Team information."""
    id: str
    name: str
    abbreviation: str
    logo_url: Optional[str] = None
    record: Optional[str] = None  # "15-10"


@dataclass
class Game:
    """Game/match information."""
    id: str
    league: str
    home_team: Team
    away_team: Team
    start_time: datetime
    status: str  # "scheduled", "in_progress", "final"
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    venue: Optional[str] = None


@dataclass
class GameOdds:
    """Betting odds for a game."""
    game_id: str
    home_team: str
    away_team: str
    home_moneyline: Optional[int] = None  # American odds (+150, -200)
    away_moneyline: Optional[int] = None
    spread: Optional[float] = None  # Point spread
    spread_odds: Optional[int] = None
    total: Optional[float] = None  # Over/under
    home_implied_prob: Optional[float] = None
    away_implied_prob: Optional[float] = None


class SportsModule:
    """Research module for sports predictions."""
    
    # ESPN API endpoints (unofficial)
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    
    LEAGUES = {
        "nba": {"sport": "basketball", "league": "nba", "name": "NBA"},
        "nfl": {"sport": "football", "league": "nfl", "name": "NFL"},
        "mlb": {"sport": "baseball", "league": "mlb", "name": "MLB"},
        "nhl": {"sport": "hockey", "league": "nhl", "name": "NHL"},
        "ncaab": {"sport": "basketball", "league": "mens-college-basketball", "name": "NCAAB"},
        "ncaaf": {"sport": "football", "league": "college-football", "name": "NCAAF"},
        "mls": {"sport": "soccer", "league": "usa.1", "name": "MLS"},
    }
    
    def __init__(self, odds_api_key: str = None):
        self.odds_api_key = odds_api_key
        self._cache: dict = {}
        self._cache_ttl = timedelta(minutes=5)
    
    async def _espn_request(self, endpoint: str) -> dict:
        """Make request to ESPN API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ESPN_BASE}{endpoint}",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    def _parse_team(self, team_data: dict) -> Team:
        """Parse ESPN team data."""
        team = team_data.get("team", team_data)
        return Team(
            id=team.get("id", ""),
            name=team.get("displayName", team.get("name", "")),
            abbreviation=team.get("abbreviation", ""),
            logo_url=team.get("logo"),
            record=None,  # Would need separate call
        )
    
    def _parse_game(self, event: dict, league: str) -> Game:
        """Parse ESPN event data into Game."""
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        
        home_data = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away_data = next((c for c in competitors if c.get("homeAway") == "away"), {})
        
        home_team = self._parse_team(home_data)
        away_team = self._parse_team(away_data)
        
        # Get scores
        home_score = int(home_data.get("score", 0)) if home_data.get("score") else None
        away_score = int(away_data.get("score", 0)) if away_data.get("score") else None
        
        # Parse time
        date_str = event.get("date", "")
        try:
            start_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            start_time = datetime.now()
        
        # Get status
        status_data = event.get("status", {})
        status_type = status_data.get("type", {}).get("name", "scheduled")
        
        if status_type == "STATUS_FINAL":
            status = "final"
        elif status_type == "STATUS_IN_PROGRESS":
            status = "in_progress"
        else:
            status = "scheduled"
        
        return Game(
            id=event.get("id", ""),
            league=league.upper(),
            home_team=home_team,
            away_team=away_team,
            start_time=start_time,
            status=status,
            home_score=home_score,
            away_score=away_score,
            venue=competition.get("venue", {}).get("fullName"),
        )
    
    async def get_games(self, league: str, date: str = None) -> list[Game]:
        """
        Get games for a league.
        
        Args:
            league: League code (nba, nfl, mlb, nhl, etc.)
            date: Date in YYYYMMDD format (default: today)
        """
        if league.lower() not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league}. Available: {list(self.LEAGUES.keys())}")
        
        league_info = self.LEAGUES[league.lower()]
        sport = league_info["sport"]
        league_code = league_info["league"]
        
        endpoint = f"/{sport}/{league_code}/scoreboard"
        if date:
            endpoint += f"?dates={date}"
        
        data = await self._espn_request(endpoint)
        events = data.get("events", [])
        
        return [self._parse_game(e, league) for e in events]
    
    async def get_standings(self, league: str) -> dict:
        """Get current standings for a league."""
        if league.lower() not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league}")
        
        league_info = self.LEAGUES[league.lower()]
        sport = league_info["sport"]
        league_code = league_info["league"]
        
        data = await self._espn_request(f"/{sport}/{league_code}/standings")
        
        standings = []
        for group in data.get("children", []):
            group_name = group.get("name", "")
            for team_standing in group.get("standings", {}).get("entries", []):
                team = team_standing.get("team", {})
                stats = {s["name"]: s["displayValue"] for s in team_standing.get("stats", [])}
                
                standings.append({
                    "team": team.get("displayName"),
                    "abbreviation": team.get("abbreviation"),
                    "group": group_name,
                    "wins": stats.get("wins"),
                    "losses": stats.get("losses"),
                    "win_pct": stats.get("winPercent"),
                })
        
        return {
            "league": league.upper(),
            "standings": standings,
        }
    
    def _moneyline_to_prob(self, moneyline: int) -> float:
        """Convert American moneyline odds to implied probability."""
        if moneyline is None:
            return 0.5
        
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)
    
    async def estimate_game_probability(
        self,
        league: str,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Estimate win probability for a game.
        
        Uses:
        1. Home advantage baseline (~55-60% in most sports)
        2. Record/standings comparison
        3. Betting lines if available
        """
        # Get standings for record comparison
        try:
            standings_data = await self.get_standings(league)
            standings = standings_data.get("standings", [])
        except:
            standings = []
        
        # Find team records
        home_record = None
        away_record = None
        
        for team in standings:
            team_name = team.get("team", "").lower()
            abbrev = team.get("abbreviation", "").lower()
            
            if home_team.lower() in team_name or home_team.lower() == abbrev:
                wins = int(team.get("wins", 0))
                losses = int(team.get("losses", 0))
                if wins + losses > 0:
                    home_record = wins / (wins + losses)
            
            if away_team.lower() in team_name or away_team.lower() == abbrev:
                wins = int(team.get("wins", 0))
                losses = int(team.get("losses", 0))
                if wins + losses > 0:
                    away_record = wins / (wins + losses)
        
        # Calculate probability
        # Base: 55% home advantage
        base_home_prob = 0.55
        
        # Adjust based on records
        if home_record and away_record:
            record_diff = home_record - away_record
            # Each 10% win rate difference = ~5% probability shift
            adjustment = record_diff * 0.5
            home_prob = max(0.1, min(0.9, base_home_prob + adjustment))
        else:
            home_prob = base_home_prob
        
        away_prob = 1 - home_prob
        
        # Confidence based on data availability
        if home_record and away_record:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "league": league.upper(),
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": round(home_prob, 2),
            "away_win_probability": round(away_prob, 2),
            "home_win_display": f"{home_prob*100:.0f}%",
            "away_win_display": f"{away_prob*100:.0f}%",
            "confidence": confidence,
            "factors": {
                "home_advantage": "55% baseline",
                "home_record": f"{home_record*100:.0f}%" if home_record else "unknown",
                "away_record": f"{away_record*100:.0f}%" if away_record else "unknown",
            },
        }
    
    async def get_sports_dashboard(self, league: str = "nba") -> dict:
        """Get sports dashboard for a league."""
        games = await self.get_games(league)
        
        return {
            "league": league.upper(),
            "games_today": len(games),
            "games": [
                {
                    "id": g.id,
                    "home": g.home_team.name,
                    "away": g.away_team.name,
                    "time": g.start_time.isoformat(),
                    "status": g.status,
                    "score": f"{g.away_score}-{g.home_score}" if g.home_score is not None else None,
                    "venue": g.venue,
                }
                for g in games
            ],
            "available_leagues": list(self.LEAGUES.keys()),
        }


# Singleton
_module: SportsModule | None = None

def get_sports_module(odds_api_key: str = None) -> SportsModule:
    global _module
    if _module is None:
        _module = SportsModule(odds_api_key)
    return _module
