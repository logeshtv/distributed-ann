/**
 * Metrics Collector - Training metrics and system monitoring
 * Tracks training progress, device performance, and system health
 */

import { Logger } from '../utils/logger.js';

export class MetricsCollector {
    constructor() {
        this.trainingHistory = [];
        this.deviceMetrics = new Map();
        this.systemMetrics = {
            serverStartTime: new Date(),
            totalRoundsCompleted: 0,
            totalSamplesProcessed: 0,
            peakDeviceCount: 0,
            avgRoundTime: 0
        };
        this.realtimeMetrics = {
            currentLoss: 0,
            currentAccuracy: 0,
            activeDevices: 0,
            samplesPerSecond: 0
        };
        this.logger = new Logger('MetricsCollector');
    }

    recordProgress(data) {
        const { deviceId, progress, batchNumber, loss, accuracy, samplesProcessed } = data;

        // Update device metrics
        const existing = this.deviceMetrics.get(deviceId) || {
            progressHistory: [],
            totalBatches: 0,
            totalSamples: 0
        };

        existing.progressHistory.push({
            progress,
            batchNumber,
            loss,
            accuracy,
            timestamp: new Date()
        });
        existing.totalBatches++;
        existing.totalSamples += samplesProcessed || 0;

        // Keep only last 100 progress entries per device
        if (existing.progressHistory.length > 100) {
            existing.progressHistory = existing.progressHistory.slice(-100);
        }

        this.deviceMetrics.set(deviceId, existing);

        // Update realtime metrics
        if (loss !== undefined) this.realtimeMetrics.currentLoss = loss;
        if (accuracy !== undefined) this.realtimeMetrics.currentAccuracy = accuracy;
    }

    recordRound(data) {
        const { round, avgLoss, avgAccuracy, devicesParticipated, timestamp } = data;

        const roundMetrics = {
            round,
            avgLoss,
            avgAccuracy,
            devicesParticipated,
            timestamp: timestamp || new Date()
        };

        this.trainingHistory.push(roundMetrics);

        // Update system metrics
        this.systemMetrics.totalRoundsCompleted = round;
        if (devicesParticipated > this.systemMetrics.peakDeviceCount) {
            this.systemMetrics.peakDeviceCount = devicesParticipated;
        }

        // Calculate average round time
        if (this.trainingHistory.length >= 2) {
            const lastTwo = this.trainingHistory.slice(-2);
            const timeDiff = new Date(lastTwo[1].timestamp) - new Date(lastTwo[0].timestamp);
            this.systemMetrics.avgRoundTime = 
                (this.systemMetrics.avgRoundTime * (round - 1) + timeDiff) / round;
        }

        // Update realtime metrics
        this.realtimeMetrics.currentLoss = avgLoss;
        this.realtimeMetrics.currentAccuracy = avgAccuracy;
        this.realtimeMetrics.activeDevices = devicesParticipated;

        this.logger.info(`Round ${round} metrics recorded: Loss=${avgLoss.toFixed(4)}, Accuracy=${avgAccuracy.toFixed(4)}`);
    }

    getMetrics() {
        return {
            system: this.systemMetrics,
            realtime: this.realtimeMetrics,
            trainingProgress: this.getTrainingProgress(),
            deviceSummary: this.getDeviceSummary()
        };
    }

    getTrainingHistory() {
        return this.trainingHistory;
    }

    getTrainingProgress() {
        if (this.trainingHistory.length === 0) return null;

        const latest = this.trainingHistory[this.trainingHistory.length - 1];
        const first = this.trainingHistory[0];

        return {
            currentRound: latest.round,
            currentLoss: latest.avgLoss,
            currentAccuracy: latest.avgAccuracy,
            lossImprovement: first.avgLoss - latest.avgLoss,
            accuracyImprovement: latest.avgAccuracy - first.avgAccuracy,
            bestLoss: Math.min(...this.trainingHistory.map(h => h.avgLoss)),
            bestAccuracy: Math.max(...this.trainingHistory.map(h => h.avgAccuracy)),
            roundsCompleted: this.trainingHistory.length
        };
    }

    getDeviceSummary() {
        const devices = Array.from(this.deviceMetrics.entries());
        return {
            totalDevices: devices.length,
            totalBatches: devices.reduce((sum, [_, d]) => sum + d.totalBatches, 0),
            totalSamples: devices.reduce((sum, [_, d]) => sum + d.totalSamples, 0),
            avgBatchesPerDevice: devices.length > 0 
                ? devices.reduce((sum, [_, d]) => sum + d.totalBatches, 0) / devices.length 
                : 0
        };
    }

    getDeviceMetrics(deviceId) {
        return this.deviceMetrics.get(deviceId) || null;
    }

    getLossHistory() {
        return this.trainingHistory.map(h => ({
            round: h.round,
            loss: h.avgLoss,
            timestamp: h.timestamp
        }));
    }

    getAccuracyHistory() {
        return this.trainingHistory.map(h => ({
            round: h.round,
            accuracy: h.avgAccuracy,
            timestamp: h.timestamp
        }));
    }

    getRecentMetrics(n = 10) {
        return this.trainingHistory.slice(-n);
    }

    clearDeviceMetrics(deviceId) {
        this.deviceMetrics.delete(deviceId);
    }

    reset() {
        this.trainingHistory = [];
        this.deviceMetrics.clear();
        this.systemMetrics.totalRoundsCompleted = 0;
        this.systemMetrics.totalSamplesProcessed = 0;
        this.realtimeMetrics = {
            currentLoss: 0,
            currentAccuracy: 0,
            activeDevices: 0,
            samplesPerSecond: 0
        };
        this.logger.info('Metrics collector reset');
    }

    exportMetrics() {
        return {
            systemMetrics: this.systemMetrics,
            trainingHistory: this.trainingHistory,
            deviceMetrics: Object.fromEntries(this.deviceMetrics),
            exportedAt: new Date()
        };
    }

    importMetrics(data) {
        if (data.systemMetrics) this.systemMetrics = data.systemMetrics;
        if (data.trainingHistory) this.trainingHistory = data.trainingHistory;
        if (data.deviceMetrics) {
            this.deviceMetrics = new Map(Object.entries(data.deviceMetrics));
        }
        this.logger.info('Metrics imported successfully');
    }
}
