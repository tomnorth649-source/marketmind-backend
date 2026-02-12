"""Arb Alert System — Push notifications when spreads open up.

Monitors markets continuously and alerts when opportunities appear.
"""
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path

from app.services.arb_scanner import get_arb_scanner, ArbOpportunity, OpportunityTier


@dataclass
class AlertConfig:
    """Alert configuration."""
    min_spread: float = 0.03  # 3% minimum spread to alert
    min_confidence: float = 0.6  # 60% match confidence
    cooldown_minutes: int = 30  # Don't re-alert same opportunity within this time
    tiers: list[str] = field(default_factory=lambda: ["hot", "warm"])


@dataclass
class Alert:
    """An alert that was triggered."""
    opportunity: ArbOpportunity
    triggered_at: datetime
    notified: bool = False
    

class ArbAlertService:
    """Service to monitor for arb opportunities and send alerts."""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.scanner = get_arb_scanner()
        self.recent_alerts: dict[str, datetime] = {}  # opportunity_key -> last_alerted
        self.callbacks: list[Callable[[Alert], None]] = []
        self._running = False
        self._state_file = Path("arb_alert_state.json")
        self._load_state()
    
    def _opportunity_key(self, opp: ArbOpportunity) -> str:
        """Generate unique key for an opportunity."""
        return f"{opp.kalshi_market.id}:{opp.poly_market.id}"
    
    def _load_state(self):
        """Load recent alerts from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self.recent_alerts = {
                    k: datetime.fromisoformat(v) 
                    for k, v in data.get("recent_alerts", {}).items()
                }
            except Exception:
                self.recent_alerts = {}
    
    def _save_state(self):
        """Save state to disk."""
        data = {
            "recent_alerts": {
                k: v.isoformat() for k, v in self.recent_alerts.items()
            }
        }
        self._state_file.write_text(json.dumps(data))
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for when alerts trigger."""
        self.callbacks.append(callback)
    
    def _should_alert(self, opp: ArbOpportunity) -> bool:
        """Check if we should alert for this opportunity."""
        # Check spread threshold
        if opp.spread < self.config.min_spread:
            return False
        
        # Check confidence threshold
        if opp.match_confidence < self.config.min_confidence:
            return False
        
        # Check tier
        if opp.tier.value not in self.config.tiers:
            return False
        
        # Check cooldown
        key = self._opportunity_key(opp)
        if key in self.recent_alerts:
            last_alert = self.recent_alerts[key]
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            if datetime.now() - last_alert < cooldown:
                return False
        
        return True
    
    async def check_once(self) -> list[Alert]:
        """Run one check and return any new alerts."""
        opportunities = await self.scanner.scan(
            min_spread=self.config.min_spread,
            min_confidence=self.config.min_confidence,
        )
        
        alerts = []
        for opp in opportunities:
            if self._should_alert(opp):
                alert = Alert(opportunity=opp, triggered_at=datetime.now())
                alerts.append(alert)
                
                # Update cooldown
                key = self._opportunity_key(opp)
                self.recent_alerts[key] = datetime.now()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Alert callback error: {e}")
        
        # Cleanup old entries (older than 24h)
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_alerts = {
            k: v for k, v in self.recent_alerts.items() if v > cutoff
        }
        
        self._save_state()
        return alerts
    
    async def run_continuous(self, interval_seconds: int = 60):
        """Run continuous monitoring."""
        self._running = True
        print(f"Starting arb alert monitor (checking every {interval_seconds}s)")
        
        while self._running:
            try:
                alerts = await self.check_once()
                if alerts:
                    print(f"🚨 {len(alerts)} new arb alerts!")
                    for alert in alerts:
                        opp = alert.opportunity
                        print(f"  [{opp.tier.value.upper()}] {opp.spread*100:.1f}¢ spread")
                        print(f"    {opp.kalshi_market.title[:50]}")
            except Exception as e:
                print(f"Alert check error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop(self):
        """Stop continuous monitoring."""
        self._running = False


# Singleton
_service: ArbAlertService | None = None

def get_alert_service() -> ArbAlertService:
    global _service
    if _service is None:
        _service = ArbAlertService()
    return _service


# Webhook/Push notification helpers
def format_alert_message(alert: Alert) -> str:
    """Format alert for messaging (Telegram, Discord, etc.)."""
    opp = alert.opportunity
    
    emoji = {"hot": "🔴", "warm": "🟡", "cold": "⚪"}.get(opp.tier.value, "⚪")
    
    return f"""
{emoji} **ARB ALERT** — {opp.spread*100:.1f}¢ spread ({opp.spread_pct:.1f}% edge)

**{opp.kalshi_market.title[:60]}**

📊 Kalshi: {opp.kalshi_market.yes_price*100:.0f}¢ YES
📊 Polymarket: {opp.poly_market.yes_price*100:.0f}¢ YES

💰 {opp.profit_example}

Match confidence: {opp.match_confidence:.0%}
""".strip()
