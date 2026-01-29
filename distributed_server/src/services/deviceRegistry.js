/**
 * Device Registry - In-memory device management
 * Tracks connected devices, their capabilities, and status
 */

import { Logger } from '../utils/logger.js';

export class DeviceRegistry {
    constructor() {
        this.devices = new Map();
        this.logger = new Logger('DeviceRegistry');
    }

    addDevice(device) {
        this.devices.set(device.id, device);
        this.logger.info(`Device added: ${device.name} (Total: ${this.devices.size})`);
        return device;
    }

    getDevice(deviceId) {
        return this.devices.get(deviceId);
    }

    updateDevice(deviceId, updates) {
        const device = this.devices.get(deviceId);
        if (device) {
            const updated = { ...device, ...updates };
            this.devices.set(deviceId, updated);
            return updated;
        }
        return null;
    }

    removeDevice(deviceId) {
        const device = this.devices.get(deviceId);
        if (device) {
            this.devices.delete(deviceId);
            this.logger.info(`Device removed: ${device.name} (Total: ${this.devices.size})`);
            return true;
        }
        return false;
    }

    getAllDevices() {
        return Array.from(this.devices.values());
    }

    getDeviceCount() {
        return this.devices.size;
    }

    getDevicesByStatus(status) {
        return Array.from(this.devices.values()).filter(d => d.status === status);
    }

    getDevicesByType(type) {
        return Array.from(this.devices.values()).filter(d => d.type === type);
    }

    getActiveDevices() {
        return Array.from(this.devices.values()).filter(d => 
            ['ready', 'training', 'submitted'].includes(d.status)
        );
    }

    getTotalResources() {
        const devices = this.getAllDevices();
        return {
            totalRam: devices.reduce((sum, d) => sum + (d.resources?.totalRam || 0), 0),
            availableRam: devices.reduce((sum, d) => sum + (d.resources?.availableRam || 0), 0),
            totalCores: devices.reduce((sum, d) => sum + (d.resources?.cpuCores || 0), 0),
            gpuDevices: devices.filter(d => d.resources?.gpuAvailable).length,
            totalDevices: devices.length,
            byType: {
                mobile: devices.filter(d => d.type === 'mobile').length,
                desktop: devices.filter(d => d.type === 'desktop').length,
                laptop: devices.filter(d => d.type === 'laptop').length,
                server: devices.filter(d => d.type === 'server').length
            }
        };
    }

    getDeviceMetrics(deviceId) {
        const device = this.devices.get(deviceId);
        if (!device) return null;

        return {
            id: device.id,
            name: device.name,
            uptime: Date.now() - new Date(device.connectedAt).getTime(),
            roundsCompleted: device.trainingStats?.roundsCompleted || 0,
            totalSamples: device.trainingStats?.totalSamples || 0,
            avgBatchTime: device.trainingStats?.avgBatchTime || 0,
            resourceUtilization: {
                ram: device.resources?.availableRam / device.resources?.totalRam || 0,
                cpu: device.resources?.cpuUsage || 0
            }
        };
    }

    clearAll() {
        this.devices.clear();
        this.logger.info('All devices cleared');
    }
}
