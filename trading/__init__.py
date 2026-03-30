from .broker import Broker, Order, Position, Account
from .paper_broker import PaperBroker
from .alpaca_broker import AlpacaBroker
from .risk_manager import RiskManager


def create_broker(settings):
    """Factory function: creates the appropriate broker based on settings.mode."""
    if settings.mode == "paper" and settings.alpaca.paper_api_key in ("YOUR_PAPER_API_KEY", ""):
        # No API key configured — use internal paper broker
        return PaperBroker(
            initial_capital=settings.trading.initial_capital,
            commission_rate=settings.trading.commission_rate,
            slippage_rate=settings.trading.slippage_rate,
        )
    elif settings.mode == "paper":
        return AlpacaBroker(settings)
    elif settings.mode == "live":
        return AlpacaBroker(settings)
    else:
        raise ValueError(f"Unknown mode: {settings.mode}")


__all__ = [
    "Broker", "Order", "Position", "Account",
    "PaperBroker", "AlpacaBroker", "RiskManager",
    "create_broker",
]
