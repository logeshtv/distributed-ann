import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';

import '../services/socket_service.dart';
import '../services/device_service.dart';
import '../services/training_service.dart';
import '../widgets/status_card.dart';
import '../widgets/metrics_display.dart';
import '../widgets/connection_panel.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Timer? _heartbeatTimer;
  final TextEditingController _serverController = TextEditingController(
    text: 'http://localhost:3001',
  );
  
  @override
  void initState() {
    super.initState();
    _initializeServices();
  }
  
  Future<void> _initializeServices() async {
    final deviceService = context.read<DeviceService>();
    await deviceService.initialize();
    
    final socketService = context.read<SocketService>();
    final trainingService = context.read<TrainingService>();
    
    // Setup socket callbacks
    socketService.onRegistered = (data) {
      _startHeartbeat();
      trainingService.markReady();
      socketService.markReady();
    };
    
    socketService.onGlobalWeights = (data) {
      trainingService.loadWeights(data);
    };
    
    socketService.onTrainingStart = (data) async {
      final result = await trainingService.runTraining(
        jobId: data['jobId'] ?? '',
        round: data['round'] ?? 0,
        batchSize: data['batchSize'] ?? 8,
        epochs: data['epochs'] ?? 1,
        learningRate: data['learningRate'] ?? 0.001,
        modelConfig: Map<String, dynamic>.from(data['modelConfig'] ?? {}),
        dataPartition: Map<String, int>.from(data['dataPartition'] ?? {}),
      );
      
      socketService.submitWeights(result);
      socketService.markReady();
    };
    
    socketService.onTrainingStop = () {
      trainingService.stop();
    };
    
    trainingService.onProgressUpdate = (data) {
      socketService.sendProgress(data);
    };
  }
  
  void _startHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      final socketService = context.read<SocketService>();
      final deviceService = context.read<DeviceService>();
      
      if (socketService.isConnected) {
        socketService.sendHeartbeat(deviceService.getHeartbeatData());
      }
    });
  }
  
  Future<void> _connect() async {
    final socketService = context.read<SocketService>();
    final deviceService = context.read<DeviceService>();
    
    socketService.setServerUrl(_serverController.text);
    await socketService.connect();
    
    // Wait for connection
    await Future.delayed(const Duration(seconds: 1));
    
    if (socketService.isConnected) {
      socketService.registerDevice(deviceService.getDeviceInfo());
    }
  }
  
  void _disconnect() {
    _heartbeatTimer?.cancel();
    final socketService = context.read<SocketService>();
    final trainingService = context.read<TrainingService>();
    
    socketService.disconnect();
    trainingService.reset();
  }
  
  @override
  void dispose() {
    _heartbeatTimer?.cancel();
    _serverController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0F172A),
              Color(0xFF1E293B),
              Color(0xFF0F172A),
            ],
          ),
        ),
        child: SafeArea(
          child: CustomScrollView(
            slivers: [
              // App Bar
              SliverAppBar(
                floating: true,
                backgroundColor: Colors.transparent,
                elevation: 0,
                title: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF06B6D4), Color(0xFF8B5CF6)],
                        ),
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF06B6D4).withOpacity(0.3),
                            blurRadius: 12,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.bolt,
                        color: Colors.white,
                        size: 24,
                      ),
                    ).animate().scale(delay: 100.ms, duration: 400.ms),
                    const SizedBox(width: 12),
                    const Text(
                      'NeuroFleet',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ).animate().fadeIn(delay: 200.ms).slideX(begin: -0.2),
                  ],
                ),
                actions: [
                  Consumer<SocketService>(
                    builder: (context, socket, _) {
                      return Container(
                        margin: const EdgeInsets.only(right: 16),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: socket.isConnected
                              ? const Color(0xFF10B981).withOpacity(0.2)
                              : const Color(0xFFF43F5E).withOpacity(0.2),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: socket.isConnected
                                ? const Color(0xFF10B981).withOpacity(0.5)
                                : const Color(0xFFF43F5E).withOpacity(0.5),
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: socket.isConnected
                                    ? const Color(0xFF10B981)
                                    : const Color(0xFFF43F5E),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              socket.isConnected ? 'Connected' : 'Offline',
                              style: TextStyle(
                                color: socket.isConnected
                                    ? const Color(0xFF10B981)
                                    : const Color(0xFFF43F5E),
                                fontWeight: FontWeight.w600,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ],
              ),
              
              // Content
              SliverPadding(
                padding: const EdgeInsets.all(16),
                sliver: SliverList(
                  delegate: SliverChildListDelegate([
                    // Connection Panel
                    ConnectionPanel(
                      serverController: _serverController,
                      onConnect: _connect,
                      onDisconnect: _disconnect,
                    ).animate().fadeIn(delay: 300.ms).slideY(begin: 0.2),
                    
                    const SizedBox(height: 16),
                    
                    // Status Cards
                    const StatusCard().animate().fadeIn(delay: 400.ms).slideY(begin: 0.2),
                    
                    const SizedBox(height: 16),
                    
                    // Metrics Display
                    const MetricsDisplay().animate().fadeIn(delay: 500.ms).slideY(begin: 0.2),
                    
                    const SizedBox(height: 16),
                    
                    // Device Info
                    _buildDeviceInfo().animate().fadeIn(delay: 600.ms).slideY(begin: 0.2),
                    
                    const SizedBox(height: 100), // Bottom padding
                  ]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildDeviceInfo() {
    return Consumer<DeviceService>(
      builder: (context, device, _) {
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: const Color(0xFF8B5CF6).withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.smartphone,
                        color: Color(0xFF8B5CF6),
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'Device Information',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                _buildInfoRow('Device', device.deviceName),
                _buildInfoRow('Platform', device.platform.toUpperCase()),
                _buildInfoRow('CPU Cores', '${device.cpuCores}'),
                _buildInfoRow('Total RAM', '${(device.totalRam / 1024).toStringAsFixed(1)} GB'),
                _buildInfoRow('Max Batch Size', '${device.maxBatchSize}'),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildBatteryWidget(device),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
  
  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.grey,
              fontSize: 14,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildBatteryWidget(DeviceService device) {
    final batteryColor = device.batteryLevel > 20
        ? const Color(0xFF10B981)
        : const Color(0xFFF43F5E);
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: batteryColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: batteryColor.withOpacity(0.3),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            device.isCharging ? Icons.battery_charging_full : Icons.battery_full,
            color: batteryColor,
          ),
          const SizedBox(width: 8),
          Text(
            '${device.batteryLevel}%',
            style: TextStyle(
              color: batteryColor,
              fontWeight: FontWeight.bold,
              fontSize: 18,
            ),
          ),
          if (device.isCharging) ...[
            const SizedBox(width: 8),
            Icon(
              Icons.bolt,
              color: batteryColor,
              size: 16,
            ),
          ],
        ],
      ),
    );
  }
}
