import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/socket_service.dart';
import '../services/training_service.dart';

class StatusCard extends StatelessWidget {
  const StatusCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer2<SocketService, TrainingService>(
      builder: (context, socket, training, _) {
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
                        color: _getStatusColor(socket, training).withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Icon(
                        _getStatusIcon(socket, training),
                        color: _getStatusColor(socket, training),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Training Status',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _getStatusText(socket, training),
                            style: TextStyle(
                              color: _getStatusColor(socket, training),
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                    _buildStatusBadge(socket, training),
                  ],
                ),
                
                if (training.isTraining) ...[
                  const SizedBox(height: 20),
                  
                  // Progress Bar
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Progress',
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 14,
                            ),
                          ),
                          Text(
                            '${training.progress.toStringAsFixed(1)}%',
                            style: const TextStyle(
                              color: Color(0xFF06B6D4),
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: training.progress / 100,
                          backgroundColor: Colors.white.withOpacity(0.1),
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            Color(0xFF06B6D4),
                          ),
                          minHeight: 8,
                        ),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Round Info
                  Row(
                    children: [
                      Expanded(
                        child: _buildStatItem(
                          'Round',
                          '${training.currentRound}',
                          const Color(0xFF8B5CF6),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildStatItem(
                          'Loss',
                          training.currentLoss.toStringAsFixed(4),
                          const Color(0xFFF43F5E),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildStatItem(
                          'Accuracy',
                          '${(training.currentAccuracy * 100).toStringAsFixed(1)}%',
                          const Color(0xFF10B981),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
  
  Color _getStatusColor(SocketService socket, TrainingService training) {
    if (!socket.isConnected) return const Color(0xFFF43F5E);
    
    switch (training.status) {
      case TrainingStatus.training:
        return const Color(0xFF06B6D4);
      case TrainingStatus.ready:
        return const Color(0xFF10B981);
      case TrainingStatus.submitted:
        return const Color(0xFF8B5CF6);
      case TrainingStatus.error:
        return const Color(0xFFF43F5E);
      default:
        return Colors.grey;
    }
  }
  
  IconData _getStatusIcon(SocketService socket, TrainingService training) {
    if (!socket.isConnected) return Icons.cloud_off;
    
    switch (training.status) {
      case TrainingStatus.training:
        return Icons.fitness_center;
      case TrainingStatus.ready:
        return Icons.check_circle;
      case TrainingStatus.submitted:
        return Icons.cloud_upload;
      case TrainingStatus.error:
        return Icons.error;
      default:
        return Icons.pause_circle;
    }
  }
  
  String _getStatusText(SocketService socket, TrainingService training) {
    if (!socket.isConnected) return 'Not connected to server';
    
    switch (training.status) {
      case TrainingStatus.training:
        return 'Training in progress...';
      case TrainingStatus.ready:
        return 'Ready for training';
      case TrainingStatus.submitted:
        return 'Weights submitted';
      case TrainingStatus.error:
        return 'Training error';
      default:
        return 'Idle';
    }
  }
  
  Widget _buildStatusBadge(SocketService socket, TrainingService training) {
    final color = _getStatusColor(socket, training);
    final text = training.status.name.toUpperCase();
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
  
  Widget _buildStatItem(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              color: Colors.grey,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}
