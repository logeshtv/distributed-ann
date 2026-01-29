"""Logging configuration."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: Path = Path("logs"),
    level: str = "INFO",
    rotation: str = "10 MB"
):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler - general logs
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        level=level,
        rotation=rotation,
        retention="30 days",
        compression="zip"
    )
    
    # File handler - errors only
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation=rotation,
        retention="90 days"
    )
    
    return logger
