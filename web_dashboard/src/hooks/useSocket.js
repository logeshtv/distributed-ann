import { useState, useEffect, useCallback } from 'react';
import { socketService } from '../services/socket';

export function useSocket() {
  const [connected, setConnected] = useState(false);
  const [devices, setDevices] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);

  useEffect(() => {
    // Connect to socket
    socketService.connect();

    // Connection state
    const unsubConnection = socketService.on('connectionChange', ({ connected }) => {
      setConnected(connected);
    });

    // Initial state from server
    const unsubState = socketService.on('dashboardState', (data) => {
      setDevices(data.devices || []);
      setJobs(data.jobs || []);
      setMetrics(data.metrics || null);
      setHistory(data.history || []);
      setCurrentRound(data.currentRound || 0);
    });

    // Device events
    const unsubDeviceConnected = socketService.on('deviceConnected', ({ device, totalDevices }) => {
      setDevices(prev => {
        const exists = prev.find(d => d.id === device.id);
        if (exists) {
          return prev.map(d => d.id === device.id ? device : d);
        }
        return [...prev, device];
      });
    });

    const unsubDeviceDisconnected = socketService.on('deviceDisconnected', ({ deviceId }) => {
      setDevices(prev => prev.filter(d => d.id !== deviceId));
    });

    const unsubDeviceResources = socketService.on('deviceResources', ({ deviceId, resources }) => {
      setDevices(prev => prev.map(d => 
        d.id === deviceId ? { ...d, resources: { ...d.resources, ...resources } } : d
      ));
    });

    // Training events
    const unsubProgress = socketService.on('trainingProgress', (data) => {
      setDevices(prev => prev.map(d =>
        d.id === data.deviceId 
          ? { ...d, status: 'training', currentProgress: data.progress }
          : d
      ));
    });

    const unsubRoundCompleted = socketService.on('roundCompleted', (data) => {
      setCurrentRound(data.round);
      setHistory(prev => [...prev, {
        round: data.round,
        avgLoss: data.avgLoss,
        avgAccuracy: data.avgAccuracy,
        devicesParticipated: data.devicesParticipated,
        timestamp: new Date()
      }]);
    });

    const unsubWeightsReceived = socketService.on('weightsReceived', (data) => {
      setDevices(prev => prev.map(d =>
        d.id === data.deviceId ? { ...d, status: 'submitted' } : d
      ));
    });

    // Job events
    const unsubJobStarted = socketService.on('jobStarted', ({ job }) => {
      setJobs(prev => prev.map(j => j.id === job.id ? job : j));
    });

    const unsubJobStopped = socketService.on('jobStopped', ({ jobId }) => {
      setJobs(prev => prev.map(j => 
        j.id === jobId ? { ...j, status: 'stopped' } : j
      ));
    });

    return () => {
      unsubConnection();
      unsubState();
      unsubDeviceConnected();
      unsubDeviceDisconnected();
      unsubDeviceResources();
      unsubProgress();
      unsubRoundCompleted();
      unsubWeightsReceived();
      unsubJobStarted();
      unsubJobStopped();
    };
  }, []);

  const refreshDevices = useCallback(() => {
    // Trigger a refresh by reconnecting to dashboard
    if (socketService.socket) {
      socketService.socket.emit('dashboard:connect');
    }
  }, []);

  return {
    connected,
    devices,
    jobs,
    metrics,
    history,
    currentRound,
    refreshDevices,
    setJobs
  };
}
