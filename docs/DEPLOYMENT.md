# Distributed Neural Network Training System - Deployment Guide

## Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for development)
- Python 3.10+ (for device clients)
- Flutter 3.0+ (for mobile app)

### 2. Environment Setup

```bash
# Clone and navigate to project
cd trading_ml_system

# Copy environment files
cp distributed_server/.env.example distributed_server/.env
```

### 3. Start with Docker

```bash
# Build and run all services
docker-compose -f docker-compose.distributed.yml up --build

# Or run in background
docker-compose -f docker-compose.distributed.yml up -d --build
```

### 4. Access Services
- **Dashboard**: http://localhost:3000
- **Server API**: http://localhost:3001
- **MongoDB**: localhost:27017

## Development Setup

### Server (Node.js)

```bash
cd distributed_server
npm install
npm run dev
```

### Dashboard (React)

```bash
cd web_dashboard
npm install
npm run dev
```

### Python Device Client

```bash
cd device_client
pip install -r requirements.txt
python src/client.py --server http://localhost:3001
```

### Flutter Mobile App

```bash
cd mobile_app
flutter pub get
flutter run
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED TRAINING SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐  WebSocket  ┌──────────────────────────────┐    │
│  │  Mobile   │◄───────────►│                              │    │
│  │  Device   │             │    CENTRAL SERVER            │    │
│  └───────────┘             │    ----------------          │    │
│                            │    • Device Registry         │    │
│  ┌───────────┐  WebSocket  │    • Job Scheduler          │    │
│  │  Laptop   │◄───────────►│    • Weight Aggregator      │    │
│  │  Client   │             │    • Metrics Collector      │    │
│  └───────────┘             │                              │    │
│                            └──────────────────────────────┘    │
│  ┌───────────┐                         │                       │
│  │  Desktop  │◄────────────────────────┘                       │
│  │  Client   │                                                 │
│  └───────────┘                                                 │
│                            ┌──────────────────────────────┐    │
│                            │      WEB DASHBOARD           │    │
│                            │      ---------------         │    │
│                            │      • Real-time Monitoring  │    │
│                            │      • Device Management     │    │
│                            │      • Training Control      │    │
│                            └──────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Training Flow

1. **Device Registration**: Clients connect and report capabilities
2. **Job Creation**: Create training job via dashboard
3. **Round Execution**: Server distributes training tasks
4. **Weight Collection**: Devices submit trained weights
5. **Aggregation**: FedAvg combines weights
6. **Broadcast**: Global model sent to all devices
7. **Repeat**: Continue until convergence

## API Reference

### REST Endpoints

```
GET  /api/devices          - List connected devices
GET  /api/jobs             - List training jobs
POST /api/jobs             - Create new job
GET  /api/jobs/:id         - Get job details
POST /api/jobs/:id/start   - Start training
POST /api/jobs/:id/stop    - Stop training
GET  /api/metrics          - Get system metrics
```

### WebSocket Events

**Client → Server:**
- `device:register` - Register device
- `device:heartbeat` - Send resource update
- `training:progress` - Report training progress
- `weights:submit` - Submit trained weights

**Server → Client:**
- `device:registered` - Registration confirmed
- `training:start` - Begin training round
- `training:stop` - Stop training
- `weights:global` - Receive aggregated weights

## Configuration

### Server Configuration

Edit `distributed_server/.env`:

```env
PORT=3001
MONGODB_URI=mongodb://localhost:27017/distributed_training
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key
MAX_DEVICES=100
HEARTBEAT_INTERVAL=30000
AGGREGATION_METHOD=fedavg
```

### Dashboard Configuration

Edit `web_dashboard/.env`:

```env
VITE_SERVER_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.distributed.yml
services:
  server:
    deploy:
      replicas: 3
```

### Load Balancing

Use nginx or HAProxy for WebSocket load balancing:

```nginx
upstream websocket {
    ip_hash;
    server server1:3001;
    server server2:3001;
    server server3:3001;
}
```

## Monitoring

### Prometheus Metrics

Server exposes metrics at `/metrics`:
- `devices_connected_total`
- `training_rounds_completed`
- `aggregation_time_seconds`
- `device_training_time_seconds`

### Grafana Dashboard

Import dashboard from `docs/grafana-dashboard.json`

## Security

### Authentication

Enable JWT authentication:

```env
AUTH_ENABLED=true
JWT_SECRET=your-super-secret-key
```

### SSL/TLS

For production, use HTTPS:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;
}
```

## Troubleshooting

### Connection Issues

1. Check server is running: `curl http://localhost:3001/health`
2. Verify WebSocket: Use browser dev tools
3. Check firewall settings

### Training Stalls

1. Check device logs for errors
2. Verify minimum devices connected
3. Check memory availability

### Memory Issues

1. Reduce batch size in job config
2. Enable gradient checkpointing
3. Use mixed precision training

## Performance Tuning

### Optimal Batch Sizes

| Device RAM | Recommended Batch |
|------------|------------------|
| 2-4 GB     | 4-8              |
| 4-8 GB     | 8-16             |
| 8-16 GB    | 16-32            |
| 16+ GB     | 32-64            |

### Network Optimization

- Use WebSocket compression
- Enable binary serialization
- Implement weight compression

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Open pull request

## License

MIT License - See LICENSE file for details
