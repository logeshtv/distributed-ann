/**
 * Job Scheduler - Training job management
 * Creates, tracks, and manages distributed training jobs
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../utils/logger.js';

export class JobScheduler {
    constructor() {
        this.jobs = new Map();
        this.activeJobId = null;
        this.logger = new Logger('JobScheduler');
    }

    createJob(config) {
        const job = {
            id: config.id || uuidv4(),
            name: config.name || 'Untitled Job',
            status: 'pending',
            modelConfig: config.modelConfig || {
                type: 'xlstm-transformer',
                inputSize: 60,
                hiddenSize: 256,
                numLayers: 4,
                numHeads: 8
            },
            dataConfig: config.dataConfig || {
                datasetName: 'trading_data',
                splitRatio: 0.8
            },
            trainingConfig: config.trainingConfig || {
                maxRounds: 100,
                localEpochs: 3,
                learningRate: 0.001,
                batchSize: 32,
                minDevices: 1,
                aggregationMethod: 'fedavg'
            },
            samplesPerDevice: config.samplesPerDevice || 1000,
            minDevices: config.minDevices || 1,
            maxRounds: config.maxRounds || 100,
            currentRound: 0,
            metrics: {
                startedAt: null,
                completedAt: null,
                bestLoss: Infinity,
                bestAccuracy: 0,
                roundHistory: []
            },
            createdAt: new Date(),
            updatedAt: new Date()
        };

        this.jobs.set(job.id, job);
        this.logger.info(`Job created: ${job.name} (${job.id})`);
        return job;
    }

    getJob(jobId) {
        return this.jobs.get(jobId);
    }

    getAllJobs() {
        return Array.from(this.jobs.values()).map(job => ({
            ...job,
            isActive: job.id === this.activeJobId
        }));
    }

    getActiveJob() {
        if (!this.activeJobId) return null;
        return this.jobs.get(this.activeJobId);
    }

    startJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) {
            throw new Error('Job not found');
        }

        if (this.activeJobId) {
            throw new Error('Another job is already running');
        }

        job.status = 'running';
        job.metrics.startedAt = new Date();
        job.updatedAt = new Date();
        this.activeJobId = jobId;
        this.jobs.set(jobId, job);

        this.logger.info(`Job started: ${job.name}`);
        return { success: true, job };
    }

    stopJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) {
            throw new Error('Job not found');
        }

        job.status = 'stopped';
        job.updatedAt = new Date();
        
        if (this.activeJobId === jobId) {
            this.activeJobId = null;
        }

        this.jobs.set(jobId, job);
        this.logger.info(`Job stopped: ${job.name}`);
        return { success: true, job };
    }

    pauseJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) {
            throw new Error('Job not found');
        }

        job.status = 'paused';
        job.updatedAt = new Date();
        this.jobs.set(jobId, job);

        this.logger.info(`Job paused: ${job.name}`);
        return { success: true, job };
    }

    resumeJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job || job.status !== 'paused') {
            throw new Error('Job not found or not paused');
        }

        job.status = 'running';
        job.updatedAt = new Date();
        this.activeJobId = jobId;
        this.jobs.set(jobId, job);

        this.logger.info(`Job resumed: ${job.name}`);
        return { success: true, job };
    }

    completeJob(jobId) {
        const job = this.jobs.get(jobId);
        if (!job) {
            throw new Error('Job not found');
        }

        job.status = 'completed';
        job.metrics.completedAt = new Date();
        job.updatedAt = new Date();

        if (this.activeJobId === jobId) {
            this.activeJobId = null;
        }

        this.jobs.set(jobId, job);
        this.logger.info(`Job completed: ${job.name}`);
        return { success: true, job };
    }

    updateJobMetrics(jobId, metrics) {
        const job = this.jobs.get(jobId);
        if (!job) return null;

        job.currentRound = metrics.round || job.currentRound;
        job.metrics.roundHistory.push({
            round: metrics.round,
            loss: metrics.loss,
            accuracy: metrics.accuracy,
            devicesParticipated: metrics.devicesParticipated,
            timestamp: new Date()
        });

        if (metrics.loss < job.metrics.bestLoss) {
            job.metrics.bestLoss = metrics.loss;
        }
        if (metrics.accuracy > job.metrics.bestAccuracy) {
            job.metrics.bestAccuracy = metrics.accuracy;
        }

        job.updatedAt = new Date();
        this.jobs.set(jobId, job);
        return job;
    }

    deleteJob(jobId) {
        if (this.activeJobId === jobId) {
            throw new Error('Cannot delete active job');
        }

        const deleted = this.jobs.delete(jobId);
        if (deleted) {
            this.logger.info(`Job deleted: ${jobId}`);
        }
        return deleted;
    }

    getJobsByStatus(status) {
        return Array.from(this.jobs.values()).filter(j => j.status === status);
    }

    getJobStats() {
        const jobs = Array.from(this.jobs.values());
        return {
            total: jobs.length,
            pending: jobs.filter(j => j.status === 'pending').length,
            running: jobs.filter(j => j.status === 'running').length,
            completed: jobs.filter(j => j.status === 'completed').length,
            stopped: jobs.filter(j => j.status === 'stopped').length,
            paused: jobs.filter(j => j.status === 'paused').length
        };
    }
}
