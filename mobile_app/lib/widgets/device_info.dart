import 'package:flutter/material.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'dart:io';

class DeviceInfoWidget extends StatelessWidget {
  final Map<String, dynamic>? deviceInfo;
  
  const DeviceInfoWidget({
    super.key,
    this.deviceInfo,
  });

  @override
  Widget build(BuildContext context) {
    final info = deviceInfo;
    
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
                  child: Icon(
                    _getDeviceIcon(),
                    color: const Color(0xFF8B5CF6),
                  ),
                ),
                const SizedBox(width: 12),
                const Text(
                  'Device Information',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            
            if (info != null) ...[
              _buildInfoRow(
                Icons.devices,
                'Device Type',
                info['type']?.toString().toUpperCase() ?? 'Unknown',
              ),
              _buildInfoRow(
                Icons.phone_android,
                'Platform',
                info['platform'] ?? 'Unknown',
              ),
              _buildInfoRow(
                Icons.memory,
                'Memory',
                _formatMemory(info['memory']),
              ),
              _buildInfoRow(
                Icons.developer_board,
                'CPU Cores',
                '${info['cpuCores'] ?? 'Unknown'} cores',
              ),
              if (info['batteryLevel'] != null)
                _buildInfoRow(
                  _getBatteryIcon(info['batteryLevel']),
                  'Battery',
                  '${info['batteryLevel']}%',
                ),
              _buildInfoRow(
                Icons.speed,
                'Status',
                info['status'] ?? 'Idle',
                valueColor: _getStatusColor(info['status']),
              ),
            ] else
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    children: [
                      CircularProgressIndicator(strokeWidth: 2),
                      SizedBox(height: 12),
                      Text(
                        'Gathering device information...',
                        style: TextStyle(
                          color: Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  IconData _getDeviceIcon() {
    return Icons.smartphone;
  }

  IconData _getBatteryIcon(int? level) {
    if (level == null) return Icons.battery_unknown;
    if (level > 80) return Icons.battery_full;
    if (level > 50) return Icons.battery_5_bar;
    if (level > 20) return Icons.battery_3_bar;
    return Icons.battery_1_bar;
  }

  String _formatMemory(dynamic memory) {
    if (memory == null) return 'Unknown';
    final bytes = memory is int ? memory : int.tryParse(memory.toString()) ?? 0;
    final gb = bytes / (1024 * 1024 * 1024);
    return '${gb.toStringAsFixed(1)} GB';
  }

  Color _getStatusColor(String? status) {
    switch (status?.toLowerCase()) {
      case 'training':
        return const Color(0xFFF59E0B);
      case 'connected':
        return const Color(0xFF10B981);
      case 'error':
        return const Color(0xFFF43F5E);
      default:
        return Colors.grey;
    }
  }

  Widget _buildInfoRow(IconData icon, String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(
            icon,
            size: 18,
            color: Colors.grey,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                color: Colors.grey,
              ),
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: valueColor,
            ),
          ),
        ],
      ),
    );
  }
}
