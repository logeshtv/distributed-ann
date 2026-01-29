"""FastAPI web application for ML Training Dashboard."""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from web.training_service import training_service

# Create FastAPI app
app = FastAPI(
    title="ML Training Dashboard",
    description="Web interface for training ML trading models",
    version="1.0.0"
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Request/Response models
class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 5e-4
    patience: int = 15
    sequence_length: int = 60
    data_path: str = "data_storage/raw"


class TrainingStatus(BaseModel):
    """Training status response."""
    is_training: bool
    progress: dict
    logs: list


class ModelInfo(BaseModel):
    """Model file information."""
    filename: str
    size_mb: float
    created_at: str
    path: str


class DataDownloadConfig(BaseModel):
    """Data download configuration."""
    source: str = "all"  # all, stocks, crypto
    universe: str = "medium"  # small, medium, large
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None


# Data download state
data_download_state = {
    "status": "idle",  # idle, downloading, completed, failed
    "progress": 0,
    "message": "",
    "current_symbol": "",
    "total_symbols": 0,
    "completed_symbols": 0
}


# API Routes
@app.get("/")
async def root():
    """Serve the main dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ML Training Dashboard - Frontend not found"}


@app.get("/api/status")
async def get_status():
    """Get current training status."""
    return JSONResponse({
        "is_training": training_service.is_training,
        "progress": training_service.progress.to_dict(),
        "logs": training_service.get_logs()
    })


@app.post("/api/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start model training."""
    if training_service.is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Validate data path
    data_path = project_root / config.data_path
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Data path not found: {config.data_path}")
    
    async def broadcast():
        await training_service.broadcast_progress()
    
    try:
        training_service.start_training(
            data_path=str(data_path),
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=config.patience,
            sequence_length=config.sequence_length,
            broadcast_callback=broadcast
        )
        return {"status": "started", "message": "Training started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train/stop")
async def stop_training():
    """Stop current training."""
    if not training_service.is_training:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    training_service.request_cancel()
    return {"status": "stopping", "message": "Cancellation requested"}


@app.get("/api/models")
async def list_models():
    """List available trained models."""
    models_dir = settings.MODELS_DIR
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            stat = model_file.stat()
            models.append({
                "filename": model_file.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(model_file)
            })
    
    # Sort by creation time (newest first)
    models.sort(key=lambda x: x["created_at"], reverse=True)
    return {"models": models}


@app.get("/api/models/{filename}/download")
async def download_model(filename: str):
    """Download a trained model."""
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    model_path = settings.MODELS_DIR / safe_filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        path=model_path,
        filename=safe_filename,
        media_type="application/octet-stream"
    )


@app.delete("/api/models/{filename}")
async def delete_model(filename: str):
    """Delete a trained model."""
    safe_filename = Path(filename).name
    model_path = settings.MODELS_DIR / safe_filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Don't allow deleting best_model.pt during training
    if training_service.is_training and safe_filename == "best_model.pt":
        raise HTTPException(status_code=400, detail="Cannot delete best model during training")
    
    model_path.unlink()
    return {"status": "deleted", "filename": safe_filename}


# WebSocket for real-time updates
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates."""
    await websocket.accept()
    training_service.websocket_clients.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "data": training_service.progress.to_dict(),
            "logs": training_service.get_logs()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages (ping/pong, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "data": training_service.progress.to_dict()
                })
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in training_service.websocket_clients:
            training_service.websocket_clients.remove(websocket)


# Data Download Endpoints
@app.get("/api/data/info")
async def get_data_info():
    """Get information about existing data files."""
    data_dir = settings.RAW_DATA_DIR
    datasets = []
    
    for subdir in ["stocks", "crypto"]:
        dir_path = data_dir / subdir
        if dir_path.exists():
            for parquet_file in dir_path.glob("*.parquet"):
                stat = parquet_file.stat()
                datasets.append({
                    "filename": parquet_file.name,
                    "type": subdir,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(parquet_file)
                })
    
    datasets.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "datasets": datasets,
        "total_size_mb": round(sum(d["size_mb"] for d in datasets), 2),
        "has_stocks": any(d["type"] == "stocks" for d in datasets),
        "has_crypto": any(d["type"] == "crypto" for d in datasets)
    }


@app.get("/api/data/status")
async def get_data_download_status():
    """Get current data download status."""
    return JSONResponse(data_download_state)


@app.post("/api/data/download")
async def start_data_download(config: DataDownloadConfig, background_tasks: BackgroundTasks):
    """Start data download in background."""
    global data_download_state
    
    if data_download_state["status"] == "downloading":
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    # Reset state
    data_download_state = {
        "status": "downloading",
        "progress": 0,
        "message": "Starting download...",
        "current_symbol": "",
        "total_symbols": 0,
        "completed_symbols": 0
    }
    
    # Start download in background
    import threading
    
    def run_download():
        global data_download_state
        try:
            import sys
            sys.path.insert(0, str(project_root))
            
            from datetime import datetime as dt
            from data.tickers import US_STOCKS, CRYPTO_PAIRS
            import yfinance as yf
            import pandas as pd
            
            # Determine symbols based on universe
            if config.universe == "small":
                stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"]
                cryptos = ["BTC-USD", "ETH-USD", "SOL-USD"]
            elif config.universe == "medium":
                stocks = US_STOCKS[:50] if len(US_STOCKS) >= 50 else US_STOCKS
                cryptos = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
                          "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD"]
            else:  # large
                stocks = US_STOCKS
                cryptos = CRYPTO_PAIRS[:50] if len(CRYPTO_PAIRS) >= 50 else CRYPTO_PAIRS
                # Convert Binance format to yfinance format
                cryptos = [c.replace("USDT", "-USD") for c in cryptos]
            
            # Parse dates
            start = dt.strptime(config.start_date, "%Y-%m-%d")
            end = dt.strptime(config.end_date, "%Y-%m-%d") if config.end_date else dt.now()
            
            all_symbols = []
            if config.source in ["all", "stocks"]:
                all_symbols.extend([("stocks", s) for s in stocks])
            if config.source in ["all", "crypto"]:
                all_symbols.extend([("crypto", s) for s in cryptos])
            
            data_download_state["total_symbols"] = len(all_symbols)
            data_download_state["message"] = f"Downloading {len(all_symbols)} symbols..."
            
            # Download data
            stocks_data = []
            crypto_data = []
            
            for i, (asset_type, symbol) in enumerate(all_symbols):
                data_download_state["current_symbol"] = symbol
                data_download_state["completed_symbols"] = i
                data_download_state["progress"] = int((i / len(all_symbols)) * 100)
                
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start, end=end, interval="1d")
                    
                    if len(df) > 0:
                        df = df.reset_index()
                        df['symbol'] = symbol
                        df = df.rename(columns={
                            'Date': 'timestamp',
                            'Datetime': 'timestamp',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        
                        # Ensure timestamp column exists
                        if 'timestamp' not in df.columns:
                            df['timestamp'] = df.index
                        
                        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                        
                        if asset_type == "stocks":
                            stocks_data.append(df)
                        else:
                            crypto_data.append(df)
                            
                except Exception as e:
                    pass  # Skip failed symbols
            
            # Save data
            if stocks_data:
                stocks_dir = settings.RAW_DATA_DIR / "stocks"
                stocks_dir.mkdir(parents=True, exist_ok=True)
                df = pd.concat(stocks_data, ignore_index=True)
                output_path = stocks_dir / f"stocks_1Day_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
                df.to_parquet(output_path, index=False)
                data_download_state["message"] = f"Saved {len(df):,} stock rows"
            
            if crypto_data:
                crypto_dir = settings.RAW_DATA_DIR / "crypto"
                crypto_dir.mkdir(parents=True, exist_ok=True)
                df = pd.concat(crypto_data, ignore_index=True)
                output_path = crypto_dir / f"crypto_1d_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
                df.to_parquet(output_path, index=False)
                data_download_state["message"] = f"Saved {len(df):,} crypto rows"
            
            data_download_state["status"] = "completed"
            data_download_state["progress"] = 100
            data_download_state["message"] = "Download complete!"
            
        except Exception as e:
            data_download_state["status"] = "failed"
            data_download_state["message"] = f"Error: {str(e)}"
    
    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    
    return {"status": "started", "message": "Download started"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)]
    )
