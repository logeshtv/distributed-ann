"""
Distributed Training Device Client
Connects to central server and participates in federated learning
"""

import asyncio
import json
import platform
import psutil
import socket
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable
import logging

import socketio
import numpy as np

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Running in simulation mode.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeviceClient')


@dataclass
class DeviceInfo:
    """Device information for registration"""
    name: str
    type: str  # 'desktop', 'laptop', 'mobile', 'server'
    platform: str
    totalRam: int  # MB - Total system RAM
    availableRam: int  # MB - Currently available RAM
    cpuCores: int  # Total CPU cores
    availableCores: int  # Available CPU cores (based on usage)
    cpuUsage: float  # Current CPU usage percentage
    gpuAvailable: bool = False
    gpuName: str = ""
    gpuMemoryTotal: int = 0  # Total GPU memory in MB
    gpuMemoryAvailable: int = 0  # Available GPU memory in MB
    gpuUtilization: float = 0.0  # GPU utilization percentage
    batteryLevel: int = 100
    isCharging: bool = True
    framework: str = 'pytorch'
    maxBatchSize: int = 32


@dataclass
class TrainingConfig:
    """Configuration for local training"""
    job_id: str
    round: int
    batch_size: int
    epochs: int
    learning_rate: float
    model_config: Dict[str, Any]
    data_partition: Dict[str, int]


class ResourceMonitor:
    """Monitors system resources"""
    
    @staticmethod
    def get_device_info(device_name: Optional[str] = None) -> DeviceInfo:
        """Gather current system information"""
        
        # Detect device type
        system = platform.system().lower()
        if system == 'darwin':
            device_type = 'laptop' if psutil.sensors_battery() else 'desktop'
            platform_name = 'macos'
        elif system == 'windows':
            device_type = 'laptop' if psutil.sensors_battery() else 'desktop'
            platform_name = 'windows'
        elif system == 'linux':
            device_type = 'server' if not psutil.sensors_battery() else 'laptop'
            platform_name = 'linux'
        else:
            device_type = 'unknown'
            platform_name = system
        
        # Memory info - AVAILABLE is what matters for training
        mem = psutil.virtual_memory()
        total_ram = mem.total // (1024 * 1024)  # Convert to MB
        available_ram = mem.available // (1024 * 1024)  # Actually available RAM
        
        # CPU info - Calculate available cores based on usage
        cpu_cores = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        # Available cores = total cores * (1 - usage%)
        available_cores = max(1, int(cpu_cores * (1 - cpu_usage / 100)))
        
        # Battery info
        battery = psutil.sensors_battery()
        battery_level = int(battery.percent) if battery else 100
        is_charging = battery.power_plugged if battery else True
        
        # GPU info - Get AVAILABLE memory, not just total
        gpu_available = False
        gpu_name = ""
        gpu_memory_total = 0
        gpu_memory_available = 0
        gpu_utilization = 0.0
        
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            
            # Get actual available GPU memory
            gpu_memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            gpu_memory_allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
            gpu_memory_available = gpu_memory_total - gpu_memory_reserved
            
            # Try to get GPU utilization via nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_utilization = float(result.stdout.strip().split('\n')[0])
            except:
                pass
        
        # Calculate max batch size based on AVAILABLE resources
        max_batch_size = ResourceMonitor.calculate_optimal_batch_size(
            available_ram, gpu_memory_available, gpu_available
        )
        
        return DeviceInfo(
            name=device_name or f"{platform_name}-{socket.gethostname()[:8]}",
            type=device_type,
            platform=platform_name,
            totalRam=total_ram,
            availableRam=available_ram,
            cpuCores=cpu_cores,
            availableCores=available_cores,
            cpuUsage=cpu_usage,
            gpuAvailable=gpu_available,
            gpuName=gpu_name,
            gpuMemoryTotal=gpu_memory_total,
            gpuMemoryAvailable=gpu_memory_available,
            gpuUtilization=gpu_utilization,
            batteryLevel=battery_level,
            isCharging=is_charging,
            framework='pytorch' if PYTORCH_AVAILABLE else 'simulation',
            maxBatchSize=max_batch_size
        )
    
    @staticmethod
    def calculate_optimal_batch_size(available_ram_mb: int, gpu_mem_mb: int = 0, use_gpu: bool = False) -> int:
        """Calculate optimal batch size based on AVAILABLE resources"""
        
        # If GPU available, use GPU memory for batch size calculation
        if use_gpu and gpu_mem_mb > 0:
            # GPU batch sizes - can be much larger
            if gpu_mem_mb >= 15000:  # 15GB+ (T4 x2, A100, etc.)
                return 512
            elif gpu_mem_mb >= 8000:  # 8GB+ (RTX 3070, T4)
                return 256
            elif gpu_mem_mb >= 4000:  # 4GB+
                return 128
            elif gpu_mem_mb >= 2000:  # 2GB+
                return 64
            else:
                return 32
        
        # CPU/RAM-based batch sizes
        if available_ram_mb >= 16000:  # 16GB+ available
            return 256
        elif available_ram_mb >= 8000:  # 8GB+
            return 128
        elif available_ram_mb >= 4000:  # 4GB+
            return 64
        elif available_ram_mb >= 2000:  # 2GB+
            return 32
        elif available_ram_mb >= 1000:  # 1GB+
            return 16
        else:
            return 8
    
    @staticmethod
    def calculate_batch_size(available_ram_mb: int) -> int:
        """Calculate optimal batch size based on available RAM (legacy)"""
        return ResourceMonitor.calculate_optimal_batch_size(available_ram_mb)
    
    @staticmethod
    def get_heartbeat_data() -> Dict[str, Any]:
        """Get current resource status for heartbeat"""
        mem = psutil.virtual_memory()
        battery = psutil.sensors_battery()
        cpu_usage = psutil.cpu_percent()
        
        # Get available resources
        available_ram = mem.available // (1024 * 1024)
        total_ram = mem.total // (1024 * 1024)
        cpu_cores = psutil.cpu_count()
        available_cores = max(1, int(cpu_cores * (1 - cpu_usage / 100)))
        
        # GPU info
        gpu_available = False
        gpu_memory_available = 0
        gpu_memory_total = 0
        gpu_utilization = 0.0
        
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            gpu_memory_available = gpu_memory_total - gpu_memory_reserved
            
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_utilization = float(result.stdout.strip().split('\n')[0])
            except:
                pass
        
        return {
            'totalRam': total_ram,
            'availableRam': available_ram,
            'ramUsagePercent': round((1 - mem.available / mem.total) * 100, 1),
            'cpuCores': cpu_cores,
            'availableCores': available_cores,
            'cpuUsage': cpu_usage,
            'gpuAvailable': gpu_available,
            'gpuMemoryTotal': gpu_memory_total,
            'gpuMemoryAvailable': gpu_memory_available,
            'gpuUtilization': gpu_utilization,
            'batteryLevel': int(battery.percent) if battery else 100,
            'isCharging': battery.power_plugged if battery else True
        }


class LocalTrainer:
    """Handles local model training"""
    
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.current_config = None
        
    def initialize_model(self, config: Dict[str, Any]):
        """Initialize or update model from config"""
        if not PYTORCH_AVAILABLE:
            logger.info("PyTorch not available, using simulation mode")
            return
            
        # Simple MLP for demonstration - replace with your model
        input_size = config.get('inputSize', 60)
        hidden_size = config.get('hiddenSize', 256)
        output_size = config.get('outputSize', 3)
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def load_weights(self, weights: Dict[str, Any]):
        """Load weights from server"""
        if not PYTORCH_AVAILABLE or self.model is None:
            return
            
        try:
            state_dict = {}
            for key, value in weights.items():
                state_dict[key] = torch.tensor(value)
            self.model.load_state_dict(state_dict)
            logger.info("Weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    
    def get_weights(self) -> Dict[str, Any]:
        """Get current model weights"""
        if not PYTORCH_AVAILABLE or self.model is None:
            # Return simulated weights
            return {f'layer_{i}': np.random.randn(100).tolist() for i in range(3)}
            
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().tolist()
        return weights
    
    async def train(
        self,
        config: TrainingConfig,
        progress_callback: Callable[[Dict[str, Any]], None]
    ) -> Dict[str, Any]:
        """Run local training"""
        
        logger.info(f"Starting training for round {config.round}")
        
        if not PYTORCH_AVAILABLE:
            # Simulation mode
            return await self._simulate_training(config, progress_callback)
        
        if self.model is None:
            self.initialize_model(config.model_config)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Generate synthetic data for demonstration
        # Replace with actual data loading
        num_samples = config.data_partition.get('end', 1000) - config.data_partition.get('start', 0)
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_times = []
        
        for epoch in range(config.epochs):
            epoch_loss = 0
            epoch_correct = 0
            
            num_batches = max(1, num_samples // config.batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = time.time()
                
                # Generate synthetic batch (replace with real data)
                x = torch.randn(config.batch_size, config.model_config.get('inputSize', 60))
                y = torch.randint(0, config.model_config.get('outputSize', 3), (config.batch_size,))
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y).sum().item()
                
                epoch_loss += loss.item()
                epoch_correct += correct
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Progress update
                progress = ((epoch * num_batches + batch_idx + 1) / (config.epochs * num_batches)) * 100
                progress_callback({
                    'progress': progress,
                    'batchNumber': batch_idx + 1,
                    'loss': loss.item(),
                    'accuracy': correct / config.batch_size
                })
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += num_batches * config.batch_size
        
        avg_loss = total_loss / (config.epochs * max(1, num_batches))
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'weights': self.get_weights(),
            'samplesProcessed': total_samples,
            'loss': avg_loss,
            'accuracy': accuracy,
            'avgBatchTime': np.mean(batch_times) if batch_times else 0,
            'round': config.round
        }
    
    async def _simulate_training(
        self,
        config: TrainingConfig,
        progress_callback: Callable[[Dict[str, Any]], None]
    ) -> Dict[str, Any]:
        """Simulate training when PyTorch is not available"""
        
        num_samples = config.data_partition.get('end', 1000) - config.data_partition.get('start', 0)
        total_batches = config.epochs * (num_samples // config.batch_size)
        
        for i in range(total_batches):
            progress = ((i + 1) / total_batches) * 100
            simulated_loss = 1.0 - (progress / 200)  # Decreasing loss
            simulated_accuracy = progress / 150  # Increasing accuracy
            
            progress_callback({
                'progress': progress,
                'batchNumber': i + 1,
                'loss': simulated_loss,
                'accuracy': min(simulated_accuracy, 0.95)
            })
            
            await asyncio.sleep(0.1)  # Simulate training time
        
        return {
            'weights': self.get_weights(),
            'samplesProcessed': num_samples,
            'loss': 0.3 + np.random.random() * 0.2,
            'accuracy': 0.7 + np.random.random() * 0.2,
            'avgBatchTime': 0.1,
            'round': config.round
        }


class DistributedClient:
    """Main client for connecting to distributed training server"""
    
    def __init__(
        self,
        server_url: str,
        device_name: Optional[str] = None,
        auto_reconnect: bool = True
    ):
        self.server_url = server_url
        self.device_name = device_name
        self.auto_reconnect = auto_reconnect
        
        self.sio = socketio.AsyncClient(
            reconnection=auto_reconnect,
            reconnection_attempts=10,
            reconnection_delay=1,
            reconnection_delay_max=30,
            logger=False
        )
        
        self.device_id = None
        self.is_training = False
        self.trainer = LocalTrainer()
        self.resource_monitor = ResourceMonitor()
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup socket event handlers"""
        
        @self.sio.event
        async def connect():
            logger.info(f"Connected to server: {self.server_url}")
            await self._register_device()
        
        @self.sio.event
        async def disconnect():
            logger.warning("Disconnected from server")
            self.is_training = False
        
        @self.sio.event
        async def connect_error(error):
            logger.error(f"Connection error: {error}")
        
        @self.sio.on('device:registered')
        async def on_registered(data):
            if data.get('success'):
                self.device_id = data.get('deviceId')
                logger.info(f"Device registered with ID: {self.device_id}")
                logger.info(f"Assigned batch size: {data.get('assignedBatchSize')}")
                # ✅ Immediately mark as ready for training
                await self.sio.emit('training:ready', {})
                logger.info("Marked as ready for training")
            else:
                logger.error("Device registration failed")
        
        @self.sio.on('device:error')
        async def on_error(data):
            logger.error(f"Device error: {data.get('error')}")
        
        @self.sio.on('weights:global')
        async def on_global_weights(data):
            logger.info(f"Received global weights for round {data.get('round')}")
            self.trainer.load_weights(data.get('weights', {}))
        
        @self.sio.on('training:start')
        async def on_training_start(data):
            logger.info("=" * 60)
            logger.info(f"🚀 TRAINING STARTED")
            logger.info(f"   Job ID: {data.get('jobId')}")
            logger.info(f"   Round: {data.get('round')}")
            logger.info(f"   Batch Size: {data.get('batchSize')}")
            logger.info(f"   Epochs: {data.get('epochs')}")
            logger.info(f"   Learning Rate: {data.get('learningRate')}")
            logger.info("=" * 60)
            
            config = TrainingConfig(
                job_id=data.get('jobId'),
                round=data.get('round'),
                batch_size=data.get('batchSize', 32),
                epochs=data.get('epochs', 1),
                learning_rate=data.get('learningRate', 0.001),
                model_config=data.get('modelConfig', {}),
                data_partition=data.get('dataPartition', {'start': 0, 'end': 1000})
            )
            
            await self._run_training(config)
        
        @self.sio.on('training:stop')
        async def on_training_stop(data):
            logger.info("Training stopped by server")
            self.is_training = False
        
        @self.sio.on('device:force_disconnect')
        async def on_force_disconnect(data):
            logger.warning(f"Force disconnected: {data.get('reason')}")
            await self.disconnect()
    
    async def _register_device(self):
        """Register this device with the server"""
        device_info = self.resource_monitor.get_device_info(self.device_name)
        await self.sio.emit('device:register', asdict(device_info))
    
    async def _send_heartbeat(self):
        """Send heartbeat with current resources"""
        while self.sio.connected:
            heartbeat_data = self.resource_monitor.get_heartbeat_data()
            await self.sio.emit('device:heartbeat', heartbeat_data)
            
            # Log current resources for visibility
            logger.info(
                f"📊 Resources - RAM: {heartbeat_data['availableRam']}MB available | "
                f"CPU: {heartbeat_data['cpuUsage']:.1f}% | "
                f"Battery: {heartbeat_data['batteryLevel']}% | "
                f"Training: {'Yes' if self.is_training else 'No'}"
            )
            await asyncio.sleep(10)  # Send every 10 seconds for better updates
    
    async def _run_training(self, config: TrainingConfig):
        """Execute local training"""
        self.is_training = True
        
        def progress_callback(data: Dict[str, Any]):
            asyncio.create_task(
                self.sio.emit('training:progress', data)
            )
        
        try:
            result = await self.trainer.train(config, progress_callback)
            
            # Submit weights to server
            await self.sio.emit('weights:submit', result)
            logger.info(f"Submitted weights for round {config.round}")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            await self.sio.emit('training:error', {'error': str(e)})
        
        finally:
            self.is_training = False
            # Mark as ready for next round
            await self.sio.emit('training:ready', {})
    
    async def connect(self):
        """Connect to the server"""
        logger.info(f"Connecting to {self.server_url}...")
        await self.sio.connect(
            self.server_url,
            transports=['websocket', 'polling']
        )
        
        # Start heartbeat task
        asyncio.create_task(self._send_heartbeat())
    
    async def disconnect(self):
        """Disconnect from the server"""
        await self.sio.disconnect()
    
    async def run_forever(self):
        """Run the client until interrupted"""
        await self.connect()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.disconnect()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Training Device Client')
    parser.add_argument('--server', '-s', default='http://localhost:3001',
                       help='Server URL (default: http://localhost:3001)')
    parser.add_argument('--name', '-n', default=None,
                       help='Device name (default: auto-generated)')
    parser.add_argument('--no-reconnect', action='store_true',
                       help='Disable auto-reconnect')
    
    args = parser.parse_args()
    
    client = DistributedClient(
        server_url=args.server,
        device_name=args.name,
        auto_reconnect=not args.no_reconnect
    )
    
    await client.run_forever()


if __name__ == '__main__':
    asyncio.run(main())
