/**
 * Weight Aggregator - Federated Averaging (FedAvg) Implementation
 * Aggregates model weights from multiple devices using weighted averaging
 */

import { Logger } from '../utils/logger.js';

export class WeightAggregator {
    constructor() {
        this.pendingWeights = new Map();
        this.globalWeights = null;
        this.currentRound = 0;
        this.aggregationHistory = [];
        this.logger = new Logger('WeightAggregator');
    }

    addWeights(submission) {
        const { deviceId, weights, samplesProcessed, loss, accuracy, round } = submission;

        this.pendingWeights.set(deviceId, {
            weights,
            samplesProcessed,
            loss,
            accuracy,
            round,
            timestamp: new Date()
        });

        this.logger.info(`Received weights from device ${deviceId} (${samplesProcessed} samples)`);
        return this.pendingWeights.size;
    }

    async aggregate() {
        if (this.pendingWeights.size === 0) {
            throw new Error('No weights to aggregate');
        }

        const submissions = Array.from(this.pendingWeights.values());
        const totalSamples = submissions.reduce((sum, s) => sum + s.samplesProcessed, 0);

        this.logger.info(`Aggregating ${submissions.length} submissions (${totalSamples} total samples)`);

        // Federated Averaging
        let aggregatedWeights = null;

        for (const submission of submissions) {
            const weight = submission.samplesProcessed / totalSamples;

            if (aggregatedWeights === null) {
                // Initialize with first submission's structure
                aggregatedWeights = this.scaleWeights(submission.weights, weight);
            } else {
                // Add weighted contribution
                aggregatedWeights = this.addWeights2(
                    aggregatedWeights,
                    this.scaleWeights(submission.weights, weight)
                );
            }
        }

        // Calculate average metrics
        const avgLoss = submissions.reduce((sum, s) => sum + s.loss * (s.samplesProcessed / totalSamples), 0);
        const avgAccuracy = submissions.reduce((sum, s) => sum + s.accuracy * (s.samplesProcessed / totalSamples), 0);

        // Update state
        this.currentRound++;
        this.globalWeights = aggregatedWeights;

        // Record history
        this.aggregationHistory.push({
            round: this.currentRound,
            devicesCount: submissions.length,
            totalSamples,
            avgLoss,
            avgAccuracy,
            timestamp: new Date()
        });

        // Clear pending weights
        this.pendingWeights.clear();

        this.logger.info(`Round ${this.currentRound} aggregation complete. Avg Loss: ${avgLoss.toFixed(4)}, Avg Accuracy: ${avgAccuracy.toFixed(4)}`);

        return {
            weights: aggregatedWeights,
            avgLoss,
            avgAccuracy,
            devicesCount: submissions.length,
            totalSamples,
            round: this.currentRound
        };
    }

    scaleWeights(weights, scale) {
        if (typeof weights === 'number') {
            return weights * scale;
        }
        
        if (Array.isArray(weights)) {
            return weights.map(w => this.scaleWeights(w, scale));
        }
        
        if (typeof weights === 'object' && weights !== null) {
            const scaled = {};
            for (const key in weights) {
                scaled[key] = this.scaleWeights(weights[key], scale);
            }
            return scaled;
        }
        
        return weights;
    }

    addWeights2(weights1, weights2) {
        if (typeof weights1 === 'number') {
            return weights1 + weights2;
        }
        
        if (Array.isArray(weights1)) {
            return weights1.map((w, i) => this.addWeights2(w, weights2[i]));
        }
        
        if (typeof weights1 === 'object' && weights1 !== null) {
            const result = {};
            for (const key in weights1) {
                result[key] = this.addWeights2(weights1[key], weights2[key]);
            }
            return result;
        }
        
        return weights1;
    }

    getGlobalWeights() {
        return this.globalWeights;
    }

    getCurrentRound() {
        return this.currentRound;
    }

    getPendingCount() {
        return this.pendingWeights.size;
    }

    getAggregationHistory() {
        return this.aggregationHistory;
    }

    getLatestAggregation() {
        return this.aggregationHistory[this.aggregationHistory.length - 1] || null;
    }

    reset() {
        this.pendingWeights.clear();
        this.globalWeights = null;
        this.currentRound = 0;
        this.aggregationHistory = [];
        this.logger.info('Aggregator reset');
    }

    getStats() {
        return {
            currentRound: this.currentRound,
            pendingSubmissions: this.pendingWeights.size,
            hasGlobalWeights: this.globalWeights !== null,
            historyLength: this.aggregationHistory.length,
            latestMetrics: this.getLatestAggregation()
        };
    }

    // Export weights in different formats
    exportWeights(format = 'json') {
        if (!this.globalWeights) return null;

        switch (format) {
            case 'json':
                return JSON.stringify(this.globalWeights);
            case 'base64':
                return Buffer.from(JSON.stringify(this.globalWeights)).toString('base64');
            default:
                return this.globalWeights;
        }
    }

    // Import weights from different formats
    importWeights(data, format = 'json') {
        switch (format) {
            case 'json':
                this.globalWeights = typeof data === 'string' ? JSON.parse(data) : data;
                break;
            case 'base64':
                this.globalWeights = JSON.parse(Buffer.from(data, 'base64').toString());
                break;
            default:
                this.globalWeights = data;
        }
        this.logger.info('Weights imported successfully');
        return true;
    }
}
