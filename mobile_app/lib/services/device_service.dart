import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:battery_plus/battery_plus.dart';

class DeviceService extends ChangeNotifier {
  final DeviceInfoPlugin _deviceInfo = DeviceInfoPlugin();
  final Battery _battery = Battery();
  
  String _deviceName = 'Unknown Device';
  String _deviceType = 'mobile';
  String _platform = 'unknown';
  int _totalRam = 0;
  int _availableRam = 0;
  int _cpuCores = 1;
  double _cpuUsage = 0;
  int _batteryLevel = 100;
  bool _isCharging = false;
  int _maxBatchSize = 4;
  
  String get deviceName => _deviceName;
  String get deviceType => _deviceType;
  String get platform => _platform;
  int get totalRam => _totalRam;
  int get availableRam => _availableRam;
  int get cpuCores => _cpuCores;
  double get cpuUsage => _cpuUsage;
  int get batteryLevel => _batteryLevel;
  bool get isCharging => _isCharging;
  int get maxBatchSize => _maxBatchSize;
  
  Future<void> initialize() async {
    await _loadDeviceInfo();
    await _updateBatteryInfo();
    await _updateResourceInfo();
    
    // Start periodic updates
    _startPeriodicUpdates();
  }
  
  Future<void> _loadDeviceInfo() async {
    try {
      if (Platform.isAndroid) {
        final androidInfo = await _deviceInfo.androidInfo;
        _deviceName = '${androidInfo.brand} ${androidInfo.model}';
        _platform = 'android';
        _deviceType = 'mobile';
        _cpuCores = androidInfo.supportedAbis.length;
      } else if (Platform.isIOS) {
        final iosInfo = await _deviceInfo.iosInfo;
        _deviceName = iosInfo.name;
        _platform = 'ios';
        _deviceType = 'mobile';
        _cpuCores = 4; // iOS doesn't expose this
      }
      
      notifyListeners();
    } catch (e) {
      debugPrint('Error loading device info: $e');
    }
  }
  
  Future<void> _updateBatteryInfo() async {
    try {
      _batteryLevel = await _battery.batteryLevel;
      final batteryState = await _battery.batteryState;
      _isCharging = batteryState == BatteryState.charging;
      notifyListeners();
    } catch (e) {
      debugPrint('Error updating battery info: $e');
    }
  }
  
  Future<void> _updateResourceInfo() async {
    // Estimate RAM based on device (actual values not available in Flutter)
    // These are rough estimates for mobile devices
    if (_platform == 'android') {
      _totalRam = 4096; // Assume 4GB
      _availableRam = 2048; // Assume 2GB available
    } else if (_platform == 'ios') {
      _totalRam = 3072; // Assume 3GB
      _availableRam = 1536; // Assume 1.5GB available
    }
    
    // Calculate max batch size based on available RAM
    _maxBatchSize = _calculateMaxBatchSize();
    
    notifyListeners();
  }
  
  int _calculateMaxBatchSize() {
    if (_availableRam < 500) return 2;
    if (_availableRam < 1000) return 4;
    if (_availableRam < 2000) return 8;
    if (_availableRam < 4000) return 16;
    return 32;
  }
  
  void _startPeriodicUpdates() {
    // Update battery and resources every 30 seconds
    Future.delayed(const Duration(seconds: 30), () async {
      await _updateBatteryInfo();
      await _updateResourceInfo();
      _startPeriodicUpdates();
    });
  }
  
  Map<String, dynamic> getDeviceInfo() {
    return {
      'name': _deviceName,
      'type': _deviceType,
      'platform': _platform,
      'totalRam': _totalRam,
      'availableRam': _availableRam,
      'cpuCores': _cpuCores,
      'cpuUsage': _cpuUsage,
      'gpuAvailable': false,
      'gpuMemory': 0,
      'batteryLevel': _batteryLevel,
      'isCharging': _isCharging,
      'framework': 'tflite',
      'maxBatchSize': _maxBatchSize,
      'modelTypes': ['small'],
    };
  }
  
  Map<String, dynamic> getHeartbeatData() {
    return {
      'availableRam': _availableRam,
      'cpuUsage': _cpuUsage,
      'batteryLevel': _batteryLevel,
      'isCharging': _isCharging,
    };
  }
  
  void setDeviceName(String name) {
    _deviceName = name;
    notifyListeners();
  }
}
