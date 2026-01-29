/**
 * Distributed Training Server - Main Entry Point
 * Central orchestration server for federated learning across devices
 */

import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';

import { SocketHandler } from './socket/socketHandler.js';
import { DeviceRegistry } from './services/deviceRegistry.js';
import { JobScheduler } from './services/jobScheduler.js';
import { WeightAggregator } from './services/weightAggregator.js';
import { MetricsCollector } from './services/metricsCollector.js';
import { Logger } from './utils/logger.js';

dotenv.config();

const app = express();
const server = createServer(app);
const PORT = process.env.PORT || 3001;

// Initialize logger
const logger = new Logger('Server');

// Socket.io with optimized settings
const io = new Server(server, {
    cors: {
        origin: process.env.CORS_ORIGIN || '*',
        methods: ['GET', 'POST'],
        credentials: true
    },
    pingTimeout: 60000,
    pingInterval: 25000,
    maxHttpBufferSize: 50 * 1024 * 1024, // 50MB for weight transfers
    transports: ['websocket', 'polling']
});

// Middleware
app.use(helmet({ contentSecurityPolicy: false }));
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 1000
});
app.use('/api/', limiter);

// Initialize services
const deviceRegistry = new DeviceRegistry();
const jobScheduler = new JobScheduler();
const weightAggregator = new WeightAggregator();
const metricsCollector = new MetricsCollector();
const socketHandler = new SocketHandler(io, deviceRegistry, jobScheduler, weightAggregator, metricsCollector);

// ==================== REST API Routes ====================

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        connectedDevices: deviceRegistry.getDeviceCount()
    });
});

// Get all connected devices
app.get('/api/devices', (req, res) => {
    try {
        const devices = deviceRegistry.getAllDevices();
        res.json({
            success: true,
            count: devices.length,
            devices: devices.map(d => ({
                id: d.id,
                name: d.name,
                type: d.type,
                platform: d.platform,
                status: d.status,
                resources: d.resources,
                connectedAt: d.connectedAt,
                lastHeartbeat: d.lastHeartbeat,
                trainingStats: d.trainingStats
            }))
        });
    } catch (error) {
        logger.error('Failed to get devices', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get specific device info
app.get('/api/devices/:id', (req, res) => {
    try {
        const device = deviceRegistry.getDevice(req.params.id);
        if (!device) {
            return res.status(404).json({ success: false, error: 'Device not found' });
        }
        res.json({ success: true, device });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Remove device
app.delete('/api/devices/:id', (req, res) => {
    try {
        const device = deviceRegistry.getDevice(req.params.id);
        if (!device) {
            return res.status(404).json({ success: false, error: 'Device not found' });
        }
        socketHandler.disconnectDevice(req.params.id);
        res.json({ success: true, message: 'Device disconnected' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get training jobs
app.get('/api/jobs', (req, res) => {
    try {
        const jobs = jobScheduler.getAllJobs();
        res.json({ success: true, jobs });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Create new training job
app.post('/api/jobs', (req, res) => {
    try {
        const { name, modelConfig, dataConfig, trainingConfig } = req.body;
        const job = jobScheduler.createJob({
            id: uuidv4(),
            name,
            modelConfig,
            dataConfig,
            trainingConfig,
            status: 'pending',
            createdAt: new Date()
        });
        res.json({ success: true, job });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Start training job
app.post('/api/jobs/:id/start', (req, res) => {
    try {
        const result = jobScheduler.startJob(req.params.id);
        socketHandler.broadcastJobStart(req.params.id);
        res.json({ success: true, result });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Stop training job
app.post('/api/jobs/:id/stop', (req, res) => {
    try {
        const result = jobScheduler.stopJob(req.params.id);
        socketHandler.broadcastJobStop(req.params.id);
        res.json({ success: true, result });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get system metrics
app.get('/api/metrics', (req, res) => {
    try {
        const metrics = metricsCollector.getMetrics();
        res.json({ success: true, metrics });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get aggregated model weights
app.get('/api/models/weights', (req, res) => {
    try {
        const weights = weightAggregator.getGlobalWeights();
        res.json({ success: true, weights: weights ? 'available' : null });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get training history
app.get('/api/history', (req, res) => {
    try {
        const history = metricsCollector.getTrainingHistory();
        res.json({ success: true, history });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// ==================== MongoDB Connection ====================
const connectDB = async () => {
    const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/distributed_training';
    try {
        await mongoose.connect(mongoUri, {
            maxPoolSize: 10,
            serverSelectionTimeoutMS: 5000
        });
        logger.info('MongoDB connected successfully');
    } catch (error) {
        logger.warn('MongoDB connection failed, running in memory mode', error.message);
    }
};

// ==================== Server Start ====================
const startServer = async () => {
    await connectDB();
    
    server.listen(PORT, () => {
        logger.info(`🚀 Distributed Training Server running on port ${PORT}`);
        logger.info(`📡 WebSocket ready for device connections`);
        logger.info(`📊 REST API available at http://localhost:${PORT}/api`);
    });
};

startServer().catch(error => {
    logger.error('Failed to start server', error);
    process.exit(1);
});

export { io, deviceRegistry, jobScheduler, weightAggregator, metricsCollector };
