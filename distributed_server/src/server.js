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
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

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

// ==================== Datasets API ====================

// Datasets state
let dataDownloadState = {
    status: 'idle',
    progress: 0,
    message: '',
    current_symbol: '',
    total_symbols: 0,
    completed_symbols: 0
};

// Get all datasets
app.get('/api/datasets', (req, res) => {
    try {
        // Look for data in data_storage directory
        const dataDir = process.env.DATA_DIR || '/data';
        const datasets = [];
        
        ['stocks', 'crypto'].forEach(type => {
            const typeDir = path.join(dataDir, 'raw', type);
            if (fs.existsSync(typeDir)) {
                fs.readdirSync(typeDir).forEach(file => {
                    if (file.endsWith('.parquet') || file.endsWith('.csv')) {
                        const stat = fs.statSync(path.join(typeDir, file));
                        datasets.push({
                            filename: file,
                            type: type,
                            size_mb: stat.size / (1024 * 1024),
                            created_at: stat.mtime.toISOString(),
                            path: path.join(typeDir, file)
                        });
                    }
                });
            }
        });
        
        datasets.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        res.json({ success: true, datasets });
    } catch (error) {
        logger.error('Failed to get datasets', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get download status
app.get('/api/datasets/download/status', (req, res) => {
    res.json(dataDownloadState);
});

// Start data download
app.post('/api/datasets/download', async (req, res) => {
    if (dataDownloadState.status === 'downloading') {
        return res.status(400).json({ success: false, error: 'Download already in progress' });
    }
    
    const { source, universe, startDate, endDate } = req.body;
    
    // Reset state
    dataDownloadState = {
        status: 'downloading',
        progress: 0,
        message: 'Starting Python download script...',
        current_symbol: '',
        total_symbols: 0,
        completed_symbols: 0
    };
    
    // Broadcast to dashboard
    io.to('dashboard').emit('download:started', dataDownloadState);
    res.json({ success: true, message: 'Download started' });
    
    // Call Python script to perform actual download
    const pythonScript = process.env.PYTHON_DOWNLOAD_SCRIPT || '/app/training_backend/scripts/api_download.py';
    const pythonCmd = process.env.PYTHON_CMD || 'python3';
    
    const pythonProcess = spawn(pythonCmd, [
        pythonScript,
        '--source', source,
        '--universe', universe,
        '--start-date', startDate,
        '--end-date', endDate
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        outputData += output;
        
        // Update progress messages
        if (output.includes('Downloading')) {
            dataDownloadState.message = output.trim();
            dataDownloadState.progress = 25;
            io.to('dashboard').emit('download:progress', dataDownloadState);
        } else if (output.includes('Fetched')) {
            dataDownloadState.message = output.trim();
            dataDownloadState.progress = 50;
            io.to('dashboard').emit('download:progress', dataDownloadState);
        } else if (output.includes('Saved')) {
            dataDownloadState.message = output.trim();
            dataDownloadState.progress = 90;
            io.to('dashboard').emit('download:progress', dataDownloadState);
        }
    });
    
    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
        logger.error('Python download error:', data.toString());
    });
    
    pythonProcess.on('close', (code) => {
        if (code === 0) {
            try {
                // Parse JSON output from Python script
                const jsonStart = outputData.lastIndexOf('{');
                if (jsonStart !== -1) {
                    const result = JSON.parse(outputData.substring(jsonStart));
                    if (result.success) {
                        dataDownloadState.status = 'completed';
                        dataDownloadState.progress = 100;
                        dataDownloadState.message = `Download complete! ${result.results.map(r => `${r.type}: ${r.rows} rows`).join(', ')}`;
                    } else {
                        dataDownloadState.status = 'failed';
                        dataDownloadState.message = `Download failed: ${result.error}`;
                    }
                } else {
                    dataDownloadState.status = 'completed';
                    dataDownloadState.progress = 100;
                    dataDownloadState.message = 'Download complete!';
                }
            } catch (e) {
                dataDownloadState.status = 'completed';
                dataDownloadState.progress = 100;
                dataDownloadState.message = 'Download complete!';
            }
        } else {
            dataDownloadState.status = 'failed';
            dataDownloadState.message = `Download failed with code ${code}: ${errorData}`;
        }
        io.to('dashboard').emit('download:completed', dataDownloadState);
    });
});

// Delete dataset
app.delete('/api/datasets/:type/:filename', (req, res) => {
    try {
        const { type, filename } = req.params;
        
        const dataDir = process.env.DATA_DIR || '/data';
        const filePath = path.join(dataDir, 'raw', type, filename);
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, error: 'Dataset not found' });
        }
        
        fs.unlinkSync(filePath);
        res.json({ success: true, message: 'Dataset deleted' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// ==================== Models API ====================

// Get all models
app.get('/api/models', (req, res) => {
    try {
        const modelsDir = process.env.MODELS_DIR || '/data/models';
        const models = [];
        
        if (fs.existsSync(modelsDir)) {
            fs.readdirSync(modelsDir).forEach(file => {
                if (file.endsWith('.pt') || file.endsWith('.pth') || file.endsWith('.onnx')) {
                    const stat = fs.statSync(path.join(modelsDir, file));
                    models.push({
                        filename: file,
                        size_mb: stat.size / (1024 * 1024),
                        created_at: stat.mtime.toISOString(),
                        path: path.join(modelsDir, file)
                    });
                }
            });
        }
        
        models.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        res.json({ success: true, models });
    } catch (error) {
        logger.error('Failed to get models', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Download model
app.get('/api/models/:filename/download', (req, res) => {
    try {
        const { filename } = req.params;
        
        const modelsDir = process.env.MODELS_DIR || '/data/models';
        const filePath = path.join(modelsDir, filename);
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, error: 'Model not found' });
        }
        
        res.download(filePath, filename);
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Delete model
app.delete('/api/models/:filename', (req, res) => {
    try {
        const fs = require('fs');
        const path = require('path');
        const { filename } = req.params;
        
        const modelsDir = process.env.MODELS_DIR || '/data/models';
        const filePath = path.join(modelsDir, filename);
        
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, error: 'Model not found' });
        }
        
        fs.unlinkSync(filePath);
        res.json({ success: true, message: 'Model deleted' });
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
