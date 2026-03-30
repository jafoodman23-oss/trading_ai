"""
SQLite storage layer using SQLAlchemy 2.0.
Stores market bars, trades, positions, model snapshots, and training episodes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    create_engine, select, and_, desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Bar(Base):
    """OHLCV bar data for a symbol."""
    __tablename__ = "bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }


class Trade(Base):
    """Record of an executed trade."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(String(4), nullable=False)   # "buy" or "sell"
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    pnl = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)


class Position(Base):
    """Current portfolio positions (upserted on each change)."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    qty = Column(Float, nullable=False)
    avg_entry = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ModelSnapshot(Base):
    """Record of saved model checkpoints with performance metrics."""
    __tablename__ = "model_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(64), nullable=False)
    path = Column(String(512), nullable=False)
    sharpe = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    n_trades = Column(Integer, nullable=True)
    is_best = Column(Boolean, nullable=False, default=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


class TrainingEpisode(Base):
    """Record of training / evaluation episodes."""
    __tablename__ = "training_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_num = Column(Integer, nullable=False)
    total_reward = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    n_trades = Column(Integer, nullable=True)
    total_return = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


class DataStorage:
    """
    High-level interface for all database operations.
    Uses SQLAlchemy with SQLite backend.
    """

    def __init__(self, db_path: str):
        url = f"sqlite:///{db_path}"
        self.engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # ------------------------------------------------------------------ #
    #  Bar methods                                                         #
    # ------------------------------------------------------------------ #

    def insert_bars(self, df: pd.DataFrame, symbol: str) -> int:
        """
        Insert bars from a DataFrame (columns: timestamp, open, high, low, close, volume, vwap).
        Skips rows already present (upsert by symbol+timestamp).
        Returns number of rows inserted.
        """
        if df.empty:
            return 0

        with self.Session() as session:
            # Fetch existing timestamps for this symbol to avoid duplicates
            existing = session.execute(
                select(Bar.timestamp).where(Bar.symbol == symbol)
            ).scalars().all()
            existing_ts = set(existing)

            rows = []
            for _, row in df.iterrows():
                ts = row["timestamp"] if isinstance(row["timestamp"], datetime) else pd.Timestamp(row["timestamp"]).to_pydatetime()
                if ts not in existing_ts:
                    rows.append(Bar(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        vwap=float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
                    ))

            if rows:
                session.add_all(rows)
                session.commit()

            return len(rows)

    def get_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch bars for a symbol between start and end as a DataFrame."""
        with self.Session() as session:
            stmt = select(Bar).where(Bar.symbol == symbol)
            if start:
                stmt = stmt.where(Bar.timestamp >= start)
            if end:
                stmt = stmt.where(Bar.timestamp <= end)
            stmt = stmt.order_by(Bar.timestamp)
            bars = session.execute(stmt).scalars().all()

        if not bars:
            return pd.DataFrame()

        records = [b.to_dict() for b in bars]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").drop(columns=["symbol"])
        return df

    def get_latest_bar_time(self, symbol: str) -> Optional[datetime]:
        """Return the timestamp of the most recent bar for a symbol."""
        with self.Session() as session:
            result = session.execute(
                select(Bar.timestamp)
                .where(Bar.symbol == symbol)
                .order_by(desc(Bar.timestamp))
                .limit(1)
            ).scalar_one_or_none()
        return result

    # ------------------------------------------------------------------ #
    #  Trade methods                                                       #
    # ------------------------------------------------------------------ #

    def insert_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        commission: float = 0.0,
        pnl: Optional[float] = None,
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Insert a trade record and return its database ID."""
        with self.Session() as session:
            trade = Trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                commission=commission,
                timestamp=timestamp or datetime.utcnow(),
                pnl=pnl,
                notes=notes,
            )
            session.add(trade)
            session.commit()
            return trade.id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Trade]:
        """Fetch trade records with optional filters."""
        with self.Session() as session:
            stmt = select(Trade)
            if symbol:
                stmt = stmt.where(Trade.symbol == symbol)
            if start:
                stmt = stmt.where(Trade.timestamp >= start)
            if end:
                stmt = stmt.where(Trade.timestamp <= end)
            stmt = stmt.order_by(desc(Trade.timestamp)).limit(limit)
            return session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------ #
    #  Position methods                                                    #
    # ------------------------------------------------------------------ #

    def upsert_position(
        self,
        symbol: str,
        qty: float,
        avg_entry: float,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
    ):
        """Create or update position record for a symbol."""
        with self.Session() as session:
            existing = session.execute(
                select(Position).where(Position.symbol == symbol)
            ).scalar_one_or_none()

            if existing:
                existing.qty = qty
                existing.avg_entry = avg_entry
                existing.current_price = current_price
                existing.unrealized_pnl = unrealized_pnl
                existing.updated_at = datetime.utcnow()
            else:
                session.add(Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry=avg_entry,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    updated_at=datetime.utcnow(),
                ))
            session.commit()

    def delete_position(self, symbol: str):
        """Remove a position record (when position is closed)."""
        with self.Session() as session:
            pos = session.execute(
                select(Position).where(Position.symbol == symbol)
            ).scalar_one_or_none()
            if pos:
                session.delete(pos)
                session.commit()

    def get_all_positions(self) -> List[Position]:
        """Return all open position records."""
        with self.Session() as session:
            return session.execute(select(Position)).scalars().all()

    # ------------------------------------------------------------------ #
    #  Model snapshot methods                                              #
    # ------------------------------------------------------------------ #

    def save_model_snapshot(
        self,
        version: str,
        path: str,
        sharpe: Optional[float] = None,
        win_rate: Optional[float] = None,
        total_return: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        n_trades: Optional[int] = None,
    ) -> int:
        """Save a model snapshot record and mark it as best if it has the highest Sharpe."""
        with self.Session() as session:
            # Check if new Sharpe is best
            best_sharpe = session.execute(
                select(ModelSnapshot.sharpe)
                .where(ModelSnapshot.is_best == True)
                .limit(1)
            ).scalar_one_or_none()

            is_best = False
            if sharpe is not None:
                if best_sharpe is None or sharpe > best_sharpe:
                    is_best = True
                    # Unmark previous best
                    session.execute(
                        select(ModelSnapshot).where(ModelSnapshot.is_best == True)
                    )
                    for prev in session.execute(
                        select(ModelSnapshot).where(ModelSnapshot.is_best == True)
                    ).scalars().all():
                        prev.is_best = False

            snap = ModelSnapshot(
                version=version,
                path=path,
                sharpe=sharpe,
                win_rate=win_rate,
                total_return=total_return,
                max_drawdown=max_drawdown,
                n_trades=n_trades,
                is_best=is_best,
                timestamp=datetime.utcnow(),
            )
            session.add(snap)
            session.commit()
            return snap.id

    def get_best_model(self) -> Optional[ModelSnapshot]:
        """Return the model snapshot with the highest Sharpe ratio."""
        with self.Session() as session:
            return session.execute(
                select(ModelSnapshot).where(ModelSnapshot.is_best == True).limit(1)
            ).scalar_one_or_none()

    def get_latest_snapshot(self) -> Optional[dict]:
        """Return the most recent model snapshot as a plain dict, or None."""
        with self.Session() as session:
            snap = session.execute(
                select(ModelSnapshot).order_by(desc(ModelSnapshot.timestamp)).limit(1)
            ).scalar_one_or_none()
            if snap is None:
                return None
            return {
                "version": snap.version,
                "path": snap.path,
                "sharpe": snap.sharpe,
                "win_rate": snap.win_rate,
                "total_return": snap.total_return,
                "timestamp": snap.timestamp,
            }

    def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Return the most recent trade records as plain dicts."""
        with self.Session() as session:
            trades = session.execute(
                select(Trade).order_by(desc(Trade.timestamp)).limit(limit)
            ).scalars().all()
            return [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "price": t.price,
                    "pnl": t.pnl,
                    "commission": t.commission,
                    "timestamp": t.timestamp,
                }
                for t in trades
            ]

    def get_model_history(self, limit: int = 50) -> List[ModelSnapshot]:
        """Return recent model snapshots ordered by timestamp."""
        with self.Session() as session:
            return session.execute(
                select(ModelSnapshot).order_by(desc(ModelSnapshot.timestamp)).limit(limit)
            ).scalars().all()

    # ------------------------------------------------------------------ #
    #  Training episode methods                                            #
    # ------------------------------------------------------------------ #

    def log_training_episode(
        self,
        episode_num: int,
        total_reward: Optional[float] = None,
        sharpe: Optional[float] = None,
        win_rate: Optional[float] = None,
        n_trades: Optional[int] = None,
        total_return: Optional[float] = None,
    ) -> int:
        """Log a training or evaluation episode record."""
        with self.Session() as session:
            ep = TrainingEpisode(
                episode_num=episode_num,
                total_reward=total_reward,
                sharpe=sharpe,
                win_rate=win_rate,
                n_trades=n_trades,
                total_return=total_return,
                timestamp=datetime.utcnow(),
            )
            session.add(ep)
            session.commit()
            return ep.id

    def get_recent_episodes(self, limit: int = 100) -> List[TrainingEpisode]:
        """Return the most recent training episodes."""
        with self.Session() as session:
            return session.execute(
                select(TrainingEpisode)
                .order_by(desc(TrainingEpisode.timestamp))
                .limit(limit)
            ).scalars().all()

    # ------------------------------------------------------------------ #
    #  Summary / reporting                                                 #
    # ------------------------------------------------------------------ #

    def get_pnl_summary(self) -> Dict:
        """Return aggregate PnL statistics."""
        with self.Session() as session:
            trades = session.execute(select(Trade)).scalars().all()

        if not trades:
            return {"total_pnl": 0.0, "n_trades": 0, "win_rate": 0.0}

        pnls = [t.pnl for t in trades if t.pnl is not None]
        if not pnls:
            return {"total_pnl": 0.0, "n_trades": len(trades), "win_rate": 0.0}

        wins = sum(1 for p in pnls if p > 0)
        return {
            "total_pnl": sum(pnls),
            "n_trades": len(pnls),
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "avg_win": sum(p for p in pnls if p > 0) / max(wins, 1),
            "avg_loss": sum(p for p in pnls if p <= 0) / max(len(pnls) - wins, 1),
        }
