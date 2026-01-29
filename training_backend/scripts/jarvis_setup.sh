#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting JarvisLabs Environment Setup..."

# 1. Update System & Install Utilities
echo "📦 Updating system packages..."
apt-get update && apt-get install -y git curl tmux htop

# 2. Install Python Dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create Directories
echo "ZE Creating data directories..."
mkdir -p data_storage/raw data_storage/processed data_storage/models logs

# 4. Environment Check
echo "🔍 Checking Environment..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 5. Launch Instructions
echo ""
echo "✅ Setup Complete!"
echo ""
echo "To start the dashboard, run:"
echo "---------------------------------------------------"
echo "python -m uvicorn web.app:app --host 0.0.0.0 --port 6006"
echo "---------------------------------------------------"
echo "ℹ️  Note: Port 6006 is typically exposed for TensorBoard on JarvisLabs,"
echo "    so we use it for our dashboard to be easily accessible."
echo "    Access it via the 'API Endpoint' or 'TensorBoard' link in your instance dashboard,"
echo "    or typically at http://<instance-ip>:6006"
