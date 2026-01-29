/**
 * Socket Handler - Real-time WebSocket event management
 * Handles device connections, weight distribution, and training coordination
 */

import { v4 as uuidv4 } from 'uuid';
import msgpack from 'msgpack-lite';
import { Logger } from '../utils/logger.js';

export class SocketHandler {
    constructor(io, deviceRegistry, jobScheduler, weightAggregator, metricsCollector) {
        this.io = io;
        this.deviceRegistry = deviceRegistry;
        this.jobScheduler = jobScheduler;
        this.weightAggregator = weightAggregator;
        this.metricsCollector = metricsCollector;
        this.logger = new Logger('SocketHandler');
        
        this.setupSocketHandlers();
        this.startHeartbeatMonitor();
    }

    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            this.logger.info(`New connection attempt: ${socket.id}`);

            // ==================== Device Registration ====================
            socket.on('device:register', (data) => {
                try {
                    const device = {
                        id: socket.id,
                        socketId: socket.id,
                        name: data.name || `Device-${socket.id.slice(0, 6)}`,
                        type: data.type || 'unknown', // 'mobile', 'desktop', 'laptop', 'server'
                        platform: data.platform || 'unknown', // 'ios', 'android', 'windows', 'macos', 'linux'
                        status: 'idle',
                        resources: {
                            totalRam: data.totalRam || 0,
                            availableRam: data.availableRam || 0,
                            cpuCores: data.cpuCores || 1,
                            cpuUsage: data.cpuUsage || 0,
                            gpuAvailable: data.gpuAvailable || false,
                            gpuMemory: data.gpuMemory || 0,
                            batteryLevel: data.batteryLevel || 100,
                            isCharging: data.isCharging || true
                        },
                        capabilities: {
                            maxBatchSize: data.maxBatchSize || this.calculateMaxBatchSize(data.availableRam),
                            supportedFramework: data.framework || 'pytorch',
                            modelTypes: data.modelTypes || ['small', 'medium']
                        },
                        connectedAt: new Date(),
                        lastHeartbeat: new Date(),
                        trainingStats: {
                            roundsCompleted: 0,
                            totalSamples: 0,
                            avgBatchTime: 0,
                            errors: 0
                        }
                    };

                    this.deviceRegistry.addDevice(device);
                    socket.join('devices');
                    
                    // Send registration confirmation
                    socket.emit('device:registered', {
                        success: true,
                        deviceId: device.id,
                        serverTime: new Date().toISOString(),
                        assignedBatchSize: device.capabilities.maxBatchSize
                    });

                    // Broadcast to dashboard
                    this.io.to('dashboard').emit('device:connected', {
                        device: this.sanitizeDevice(device),
                        totalDevices: this.deviceRegistry.getDeviceCount()
                    });

                    this.logger.info(`Device registered: ${device.name} (${device.type})`);

                    // Send current global weights if available
                    const globalWeights = this.weightAggregator.getGlobalWeights();
                    if (globalWeights) {
                        socket.emit('weights:global', {
                            weights: globalWeights,
                            round: this.weightAggregator.getCurrentRound()
                        });
                    }

                } catch (error) {
                    this.logger.error('Device registration failed', error);
                    socket.emit('device:error', { error: 'Registration failed' });
                }
            });

            // ==================== Heartbeat & Resources ====================
            socket.on('device:heartbeat', (data) => {
                try {
                    this.deviceRegistry.updateDevice(socket.id, {
                        lastHeartbeat: new Date(),
                        resources: {
                            ...this.deviceRegistry.getDevice(socket.id)?.resources,
                            availableRam: data.availableRam,
                            cpuUsage: data.cpuUsage,
                            batteryLevel: data.batteryLevel,
                            isCharging: data.isCharging
                        }
                    });

                    // Broadcast resource update to dashboard
                    this.io.to('dashboard').emit('device:resources', {
                        deviceId: socket.id,
                        resources: data
                    });

                } catch (error) {
                    this.logger.error('Heartbeat update failed', error);
                }
            });

            // ==================== Training Coordination ====================
            socket.on('training:ready', (data) => {
                try {
                    this.deviceRegistry.updateDevice(socket.id, {
                        status: 'ready'
                    });

                    // Check if we have enough devices to start
                    const readyDevices = this.deviceRegistry.getDevicesByStatus('ready');
                    const activeJob = this.jobScheduler.getActiveJob();

                    if (activeJob && readyDevices.length >= (activeJob.minDevices || 1)) {
                        this.distributeTrainingBatches(activeJob);
                    }

                    this.logger.info(`Device ${socket.id} ready for training`);

                } catch (error) {
                    this.logger.error('Training ready failed', error);
                }
            });

            socket.on('training:progress', (data) => {
                try {
                    const device = this.deviceRegistry.getDevice(socket.id);
                    if (device) {
                        this.deviceRegistry.updateDevice(socket.id, {
                            status: 'training',
                            currentProgress: data.progress,
                            currentBatch: data.batchNumber
                        });

                        // Update metrics
                        this.metricsCollector.recordProgress({
                            deviceId: socket.id,
                            ...data
                        });

                        // Broadcast to dashboard
                        this.io.to('dashboard').emit('training:progress', {
                            deviceId: socket.id,
                            deviceName: device.name,
                            ...data
                        });
                    }

                } catch (error) {
                    this.logger.error('Progress update failed', error);
                }
            });

            // ==================== Weight Submission ====================
            socket.on('weights:submit', async (data) => {
                try {
                    const device = this.deviceRegistry.getDevice(socket.id);
                    if (!device) return;

                    this.logger.info(`Received weights from ${device.name}`);

                    // Update device stats
                    this.deviceRegistry.updateDevice(socket.id, {
                        status: 'submitted',
                        trainingStats: {
                            ...device.trainingStats,
                            roundsCompleted: device.trainingStats.roundsCompleted + 1,
                            totalSamples: device.trainingStats.totalSamples + data.samplesProcessed,
                            avgBatchTime: data.avgBatchTime
                        }
                    });

                    // Add weights to aggregator
                    await this.weightAggregator.addWeights({
                        deviceId: socket.id,
                        weights: data.weights,
                        samplesProcessed: data.samplesProcessed,
                        loss: data.loss,
                        accuracy: data.accuracy,
                        round: data.round
                    });

                    // Check if all devices have submitted
                    const submittedCount = this.deviceRegistry.getDevicesByStatus('submitted').length;
                    const totalDevices = this.deviceRegistry.getDevicesByStatus('training').length + submittedCount;

                    if (submittedCount >= totalDevices && submittedCount > 0) {
                        await this.aggregateAndDistribute();
                    }

                    socket.emit('weights:acknowledged', { success: true });

                    // Broadcast to dashboard
                    this.io.to('dashboard').emit('weights:received', {
                        deviceId: socket.id,
                        deviceName: device.name,
                        round: data.round,
                        loss: data.loss,
                        accuracy: data.accuracy
                    });

                } catch (error) {
                    this.logger.error('Weight submission failed', error);
                    socket.emit('weights:error', { error: error.message });
                }
            });

            // ==================== Dashboard Connection ====================
            socket.on('dashboard:connect', () => {
                socket.join('dashboard');
                
                // Send current state
                socket.emit('dashboard:state', {
                    devices: this.deviceRegistry.getAllDevices().map(d => this.sanitizeDevice(d)),
                    jobs: this.jobScheduler.getAllJobs(),
                    currentRound: this.weightAggregator.getCurrentRound(),
                    metrics: this.metricsCollector.getMetrics(),
                    history: this.metricsCollector.getTrainingHistory()
                });

                this.logger.info('Dashboard connected');
            });

            // ==================== Disconnection ====================
            socket.on('disconnect', (reason) => {
                try {
                    const device = this.deviceRegistry.getDevice(socket.id);
                    if (device) {
                        this.deviceRegistry.removeDevice(socket.id);
                        
                        // Broadcast to dashboard
                        this.io.to('dashboard').emit('device:disconnected', {
                            deviceId: socket.id,
                            deviceName: device.name,
                            reason,
                            totalDevices: this.deviceRegistry.getDeviceCount()
                        });

                        this.logger.info(`Device disconnected: ${device.name} (${reason})`);
                    }

                } catch (error) {
                    this.logger.error('Disconnect handler failed', error);
                }
            });

            // ==================== Error Handling ====================
            socket.on('error', (error) => {
                this.logger.error(`Socket error for ${socket.id}`, error);
            });
        });
    }

    // ==================== Helper Methods ====================

    calculateMaxBatchSize(availableRam) {
        // Estimate batch size based on available RAM (in MB)
        if (availableRam < 500) return 4;
        if (availableRam < 1000) return 8;
        if (availableRam < 2000) return 16;
        if (availableRam < 4000) return 32;
        if (availableRam < 8000) return 64;
        return 128;
    }

    sanitizeDevice(device) {
        // Remove sensitive data before sending to dashboard
        const { socketId, ...sanitized } = device;
        return sanitized;
    }

    async distributeTrainingBatches(job) {
        const devices = this.deviceRegistry.getDevicesByStatus('ready');
        const round = this.weightAggregator.getCurrentRound() + 1;

        this.logger.info(`Distributing training batches for round ${round} to ${devices.length} devices`);

        devices.forEach((device, index) => {
            this.deviceRegistry.updateDevice(device.id, { status: 'training' });

            this.io.to(device.socketId).emit('training:start', {
                jobId: job.id,
                round,
                batchSize: device.capabilities.maxBatchSize,
                epochs: job.trainingConfig?.localEpochs || 1,
                learningRate: job.trainingConfig?.learningRate || 0.001,
                modelConfig: job.modelConfig,
                dataPartition: {
                    start: index * job.samplesPerDevice,
                    end: (index + 1) * job.samplesPerDevice
                }
            });
        });

        // Broadcast to dashboard
        this.io.to('dashboard').emit('training:round_started', {
            round,
            devices: devices.length
        });
    }

    async aggregateAndDistribute() {
        try {
            this.logger.info('Aggregating weights from all devices...');

            // Perform FedAvg aggregation
            const aggregatedWeights = await this.weightAggregator.aggregate();
            const currentRound = this.weightAggregator.getCurrentRound();

            // Record metrics
            this.metricsCollector.recordRound({
                round: currentRound,
                avgLoss: aggregatedWeights.avgLoss,
                avgAccuracy: aggregatedWeights.avgAccuracy,
                devicesParticipated: aggregatedWeights.devicesCount,
                timestamp: new Date()
            });

            // Distribute new global weights
            this.io.to('devices').emit('weights:global', {
                weights: aggregatedWeights.weights,
                round: currentRound,
                avgLoss: aggregatedWeights.avgLoss,
                avgAccuracy: aggregatedWeights.avgAccuracy
            });

            // Reset device statuses
            this.deviceRegistry.getAllDevices().forEach(device => {
                if (device.status === 'submitted') {
                    this.deviceRegistry.updateDevice(device.id, { status: 'ready' });
                }
            });

            // Broadcast to dashboard
            this.io.to('dashboard').emit('training:round_completed', {
                round: currentRound,
                avgLoss: aggregatedWeights.avgLoss,
                avgAccuracy: aggregatedWeights.avgAccuracy,
                devicesParticipated: aggregatedWeights.devicesCount
            });

            this.logger.info(`Round ${currentRound} completed. Avg Loss: ${aggregatedWeights.avgLoss.toFixed(4)}`);

            // Check if more rounds needed
            const activeJob = this.jobScheduler.getActiveJob();
            if (activeJob && currentRound < activeJob.maxRounds) {
                setTimeout(() => this.distributeTrainingBatches(activeJob), 1000);
            } else if (activeJob) {
                this.jobScheduler.completeJob(activeJob.id);
                this.io.to('dashboard').emit('training:completed', {
                    jobId: activeJob.id,
                    finalRound: currentRound,
                    finalLoss: aggregatedWeights.avgLoss,
                    finalAccuracy: aggregatedWeights.avgAccuracy
                });
            }

        } catch (error) {
            this.logger.error('Weight aggregation failed', error);
        }
    }

    disconnectDevice(deviceId) {
        const device = this.deviceRegistry.getDevice(deviceId);
        if (device) {
            this.io.to(device.socketId).emit('device:force_disconnect', {
                reason: 'Disconnected by admin'
            });
            this.io.sockets.sockets.get(device.socketId)?.disconnect(true);
        }
    }

    broadcastJobStart(jobId) {
        const job = this.jobScheduler.getJob(jobId);
        this.io.to('dashboard').emit('job:started', { job });
    }

    broadcastJobStop(jobId) {
        this.io.to('devices').emit('training:stop', { jobId });
        this.io.to('dashboard').emit('job:stopped', { jobId });
    }

    startHeartbeatMonitor() {
        setInterval(() => {
            const now = Date.now();
            const devices = this.deviceRegistry.getAllDevices();
            
            devices.forEach(device => {
                const lastHeartbeat = new Date(device.lastHeartbeat).getTime();
                if (now - lastHeartbeat > 60000) { // 60 seconds timeout
                    this.logger.warn(`Device ${device.name} timed out`);
                    this.io.sockets.sockets.get(device.socketId)?.disconnect(true);
                }
            });
        }, 30000);
    }
}
