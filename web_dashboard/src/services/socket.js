import { io } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:3001';

class SocketService {
  constructor() {
    this.socket = null;
    this.listeners = new Map();
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    if (this.socket?.connected) return;

    this.socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
      timeout: 10000
    });

    this.socket.on('connect', () => {
      console.log('🔌 Socket connected');
      this.connected = true;
      this.reconnectAttempts = 0;
      
      // Register as dashboard
      this.socket.emit('dashboard:connect');
      
      this.emit('connectionChange', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('🔌 Socket disconnected:', reason);
      this.connected = false;
      this.emit('connectionChange', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      this.reconnectAttempts++;
      this.emit('connectionError', { error, attempts: this.reconnectAttempts });
    });

    // Dashboard state events
    this.socket.on('dashboard:state', (data) => {
      this.emit('dashboardState', data);
    });

    // Device events
    this.socket.on('device:connected', (data) => {
      this.emit('deviceConnected', data);
    });

    this.socket.on('device:disconnected', (data) => {
      this.emit('deviceDisconnected', data);
    });

    this.socket.on('device:resources', (data) => {
      this.emit('deviceResources', data);
    });

    // Training events
    this.socket.on('training:progress', (data) => {
      this.emit('trainingProgress', data);
    });

    this.socket.on('training:round_started', (data) => {
      this.emit('roundStarted', data);
    });

    this.socket.on('training:round_completed', (data) => {
      this.emit('roundCompleted', data);
    });

    this.socket.on('training:completed', (data) => {
      this.emit('trainingCompleted', data);
    });

    // Weight events
    this.socket.on('weights:received', (data) => {
      this.emit('weightsReceived', data);
    });

    // Job events
    this.socket.on('job:started', (data) => {
      this.emit('jobStarted', data);
    });

    this.socket.on('job:stopped', (data) => {
      this.emit('jobStopped', data);
    });

    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    
    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }

  off(event, callback) {
    this.listeners.get(event)?.delete(callback);
  }

  emit(event, data) {
    this.listeners.get(event)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in ${event} listener:`, error);
      }
    });
  }

  isConnected() {
    return this.connected;
  }
}

export const socketService = new SocketService();
