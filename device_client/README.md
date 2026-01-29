# Distributed Training Device Client

A Python client that connects to the central training server and participates in federated learning.

## Features

- **Auto Device Detection**: Automatically detects device type, platform, and resources
- **Resource Monitoring**: Real-time monitoring of RAM, CPU, and battery
- **Local Training**: PyTorch-based local model training
- **Weight Synchronization**: Receives and submits model weights to/from the server
- **Auto Reconnect**: Automatically reconnects if connection is lost
- **Memory-Aware**: Adjusts batch size based on available resources

## Installation

```bash
cd device_client
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python src/client.py
```

### With Custom Server URL
```bash
python src/client.py --server http://your-server:3001
```

### With Custom Device Name
```bash
python src/client.py --name "MyLaptop"
```

### Full Options
```bash
python src/client.py --server http://localhost:3001 --name "TrainingDevice1" --no-reconnect
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--server` | `-s` | `http://localhost:3001` | Server URL |
| `--name` | `-n` | Auto-generated | Custom device name |
| `--no-reconnect` | | `False` | Disable auto-reconnect |

## Architecture

```
device_client/
├── src/
│   └── client.py      # Main client implementation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Classes

### `DeviceInfo`
Data class containing device information sent during registration.

### `ResourceMonitor`
Monitors system resources (RAM, CPU, Battery) for heartbeat updates.

### `LocalTrainer`
Handles local model training with PyTorch. Falls back to simulation if PyTorch is unavailable.

### `DistributedClient`
Main client class that manages connection, events, and training coordination.

## Events

The client handles the following socket events:

| Event | Direction | Description |
|-------|-----------|-------------|
| `device:register` | → Server | Register device with capabilities |
| `device:registered` | ← Server | Registration confirmation |
| `device:heartbeat` | → Server | Periodic resource update |
| `training:start` | ← Server | Start local training |
| `training:progress` | → Server | Report training progress |
| `weights:submit` | → Server | Submit trained weights |
| `weights:global` | ← Server | Receive aggregated weights |

## Simulation Mode

If PyTorch is not installed, the client runs in simulation mode:
- Generates random weights
- Simulates training progress
- Useful for testing without full ML stack

## Integration with Existing Models

To use your own model, modify the `LocalTrainer` class:

```python
class LocalTrainer:
    def initialize_model(self, config):
        # Replace with your model
        self.model = YourCustomModel(config)
    
    async def train(self, config, progress_callback):
        # Add your training logic
        pass
```
