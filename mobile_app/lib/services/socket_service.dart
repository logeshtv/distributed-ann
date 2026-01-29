import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:socket_io_client/socket_io_client.dart' as IO;

class SocketService extends ChangeNotifier {
  IO.Socket? _socket;
  bool _isConnected = false;
  String? _deviceId;
  String _serverUrl = 'http://localhost:3001';
  
  // Event callbacks
  Function(Map<String, dynamic>)? onRegistered;
  Function(Map<String, dynamic>)? onGlobalWeights;
  Function(Map<String, dynamic>)? onTrainingStart;
  Function()? onTrainingStop;
  Function(String)? onForceDisconnect;
  
  bool get isConnected => _isConnected;
  String? get deviceId => _deviceId;
  String get serverUrl => _serverUrl;
  
  void setServerUrl(String url) {
    _serverUrl = url;
    notifyListeners();
  }
  
  Future<void> connect() async {
    if (_socket != null) {
      _socket!.dispose();
    }
    
    _socket = IO.io(_serverUrl, <String, dynamic>{
      'transports': ['websocket', 'polling'],
      'autoConnect': true,
      'reconnection': true,
      'reconnectionAttempts': 10,
      'reconnectionDelay': 1000,
    });
    
    _socket!.onConnect((_) {
      debugPrint('🔌 Connected to server');
      _isConnected = true;
      notifyListeners();
    });
    
    _socket!.onDisconnect((_) {
      debugPrint('🔌 Disconnected from server');
      _isConnected = false;
      _deviceId = null;
      notifyListeners();
    });
    
    _socket!.onConnectError((error) {
      debugPrint('❌ Connection error: $error');
      _isConnected = false;
      notifyListeners();
    });
    
    // Device events
    _socket!.on('device:registered', (data) {
      final Map<String, dynamic> response = Map<String, dynamic>.from(data);
      if (response['success'] == true) {
        _deviceId = response['deviceId'];
        debugPrint('✅ Device registered: $_deviceId');
        onRegistered?.call(response);
        notifyListeners();
      }
    });
    
    _socket!.on('device:error', (data) {
      debugPrint('❌ Device error: $data');
    });
    
    // Weight events
    _socket!.on('weights:global', (data) {
      debugPrint('📦 Received global weights');
      final Map<String, dynamic> response = Map<String, dynamic>.from(data);
      onGlobalWeights?.call(response);
    });
    
    _socket!.on('weights:acknowledged', (data) {
      debugPrint('✅ Weights acknowledged');
    });
    
    // Training events
    _socket!.on('training:start', (data) {
      debugPrint('🚀 Training started');
      final Map<String, dynamic> response = Map<String, dynamic>.from(data);
      onTrainingStart?.call(response);
    });
    
    _socket!.on('training:stop', (data) {
      debugPrint('⏹️ Training stopped');
      onTrainingStop?.call();
    });
    
    _socket!.on('device:force_disconnect', (data) {
      debugPrint('⚠️ Force disconnected');
      final Map<String, dynamic> response = Map<String, dynamic>.from(data);
      onForceDisconnect?.call(response['reason'] ?? 'Unknown');
      disconnect();
    });
  }
  
  void disconnect() {
    _socket?.dispose();
    _socket = null;
    _isConnected = false;
    _deviceId = null;
    notifyListeners();
  }
  
  void emit(String event, dynamic data) {
    if (_socket != null && _isConnected) {
      _socket!.emit(event, data);
    }
  }
  
  void registerDevice(Map<String, dynamic> deviceInfo) {
    emit('device:register', deviceInfo);
  }
  
  void sendHeartbeat(Map<String, dynamic> data) {
    emit('device:heartbeat', data);
  }
  
  void sendProgress(Map<String, dynamic> data) {
    emit('training:progress', data);
  }
  
  void submitWeights(Map<String, dynamic> data) {
    emit('weights:submit', data);
  }
  
  void markReady() {
    emit('training:ready', {});
  }
  
  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}
