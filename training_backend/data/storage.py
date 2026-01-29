"""Data storage and database operations."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class OHLCVData(Base):
    """OHLCV data table model."""
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    source = Column(String(50))  # 'alpaca', 'binance', etc.


class ModelCheckpoint(Base):
    """Model checkpoint metadata."""
    __tablename__ = 'model_checkpoints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    path = Column(Text)
    metrics = Column(Text)  # JSON string
    config = Column(Text)  # JSON string


class TradeLog(Base):
    """Trade execution log."""
    __tablename__ = 'trade_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String(10))  # 'buy', 'sell'
    quantity = Column(Float)
    price = Column(Float)
    commission = Column(Float)
    slippage = Column(Float)
    realized_pnl = Column(Float, nullable=True)
    strategy = Column(String(50), nullable=True)


class DataStorage:
    """
    Data storage manager for trading system.
    
    Handles:
    - Database connections (SQLite/PostgreSQL)
    - Parquet file storage
    - OHLCV data persistence
    - Model checkpoint management
    """
    
    def __init__(
        self,
        db_url: str = "sqlite:///data_storage/trading.db",
        data_dir: Optional[Path] = None
    ):
        """
        Initialize data storage.
        
        Args:
            db_url: Database connection URL
            data_dir: Directory for file storage
        """
        self.db_url = db_url
        self.data_dir = data_dir or Path("data_storage")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database engine
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info(f"DataStorage initialized with {db_url}")
    
    def save_ohlcv_to_db(
        self,
        df: pd.DataFrame,
        source: str = "unknown"
    ) -> int:
        """
        Save OHLCV data to database.
        
        Args:
            df: DataFrame with OHLCV data
            source: Data source identifier
            
        Returns:
            Number of rows inserted
        """
        df = df.copy()
        df['source'] = source
        
        # Use pandas to_sql for bulk insert
        n_rows = df.to_sql(
            'ohlcv_data',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        logger.info(f"Saved {len(df)} rows to database")
        return len(df)
    
    def load_ohlcv_from_db(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date filter
            end_date: End date filter
            source: Source filter
            
        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM ohlcv_data WHERE 1=1"
        params = {}
        
        if symbols:
            query += " AND symbol IN :symbols"
            params['symbols'] = tuple(symbols)
        
        if start_date:
            query += " AND timestamp >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND timestamp <= :end_date"
            params['end_date'] = end_date
        
        if source:
            query += " AND source = :source"
            params['source'] = source
        
        query += " ORDER BY symbol, timestamp"
        
        df = pd.read_sql(query, self.engine, params=params)
        logger.info(f"Loaded {len(df)} rows from database")
        
        return df
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: str = "raw"
    ) -> Path:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            subdir: Subdirectory within data_dir
            
        Returns:
            Path to saved file
        """
        output_dir = self.data_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path
    
    def load_from_parquet(
        self,
        filename: str,
        subdir: str = "raw"
    ) -> pd.DataFrame:
        """
        Load DataFrame from parquet file.
        
        Args:
            filename: Input filename (with or without .parquet)
            subdir: Subdirectory within data_dir
            
        Returns:
            Loaded DataFrame
        """
        if not filename.endswith('.parquet'):
            filename = f"{filename}.parquet"
        
        input_path = self.data_dir / subdir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        return df
    
    def save_model_checkpoint(
        self,
        name: str,
        version: str,
        model_path: Path,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> int:
        """
        Save model checkpoint metadata to database.
        
        Args:
            name: Model name
            version: Model version
            model_path: Path to saved model file
            metrics: Performance metrics
            config: Model configuration
            
        Returns:
            Checkpoint ID
        """
        import json
        
        session = self.Session()
        try:
            checkpoint = ModelCheckpoint(
                name=name,
                version=version,
                path=str(model_path),
                metrics=json.dumps(metrics),
                config=json.dumps(config)
            )
            session.add(checkpoint)
            session.commit()
            checkpoint_id = checkpoint.id
            logger.info(f"Saved model checkpoint: {name} v{version}")
            return checkpoint_id
        finally:
            session.close()
    
    def get_latest_checkpoint(
        self,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest model checkpoint by name.
        
        Args:
            name: Model name
            
        Returns:
            Checkpoint metadata dict or None
        """
        import json
        
        session = self.Session()
        try:
            checkpoint = session.query(ModelCheckpoint)\
                .filter(ModelCheckpoint.name == name)\
                .order_by(ModelCheckpoint.created_at.desc())\
                .first()
            
            if checkpoint:
                return {
                    'id': checkpoint.id,
                    'name': checkpoint.name,
                    'version': checkpoint.version,
                    'created_at': checkpoint.created_at,
                    'path': checkpoint.path,
                    'metrics': json.loads(checkpoint.metrics),
                    'config': json.loads(checkpoint.config)
                }
            return None
        finally:
            session.close()
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        realized_pnl: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> int:
        """
        Log a trade execution.
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Trade quantity
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            realized_pnl: Realized P&L (for closing trades)
            strategy: Strategy name
            
        Returns:
            Trade log ID
        """
        session = self.Session()
        try:
            trade = TradeLog(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                commission=commission,
                slippage=slippage,
                realized_pnl=realized_pnl,
                strategy=strategy
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
            logger.info(f"Logged trade: {action} {quantity} {symbol} @ {price}")
            return trade_id
        finally:
            session.close()
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get trade history from database.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            strategy: Filter by strategy
            
        Returns:
            DataFrame with trade history
        """
        query = "SELECT * FROM trade_logs WHERE 1=1"
        params = {}
        
        if symbol:
            query += " AND symbol = :symbol"
            params['symbol'] = symbol
        
        if start_date:
            query += " AND timestamp >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND timestamp <= :end_date"
            params['end_date'] = end_date
        
        if strategy:
            query += " AND strategy = :strategy"
            params['strategy'] = strategy
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql(query, self.engine, params=params)
        return df
    
    def list_parquet_files(self, subdir: str = "raw") -> List[Path]:
        """List all parquet files in a subdirectory."""
        dir_path = self.data_dir / subdir
        if not dir_path.exists():
            return []
        return list(dir_path.glob("*.parquet"))
