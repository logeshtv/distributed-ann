#!/bin/bash

# ============================================================
# Distributed Neural Network Training System - Start Script
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Fancy header
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              🚀 NEUROFLEET - Distributed Training System             ║"
echo "║                                                                      ║"
echo "║    Connect your devices. Train together. Achieve the extraordinary. ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODE=${1:-"dev"}
COMPONENT=${2:-"all"}

print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking Prerequisites"
    
    local missing=0
    
    # Check Node.js
    if command -v node &> /dev/null; then
        print_success "Node.js $(node --version) found"
    else
        print_error "Node.js not found"
        missing=1
    fi
    
    # Check npm
    if command -v npm &> /dev/null; then
        print_success "npm $(npm --version) found"
    else
        print_error "npm not found"
        missing=1
    fi
    
    # Check Docker (optional for dev)
    if command -v docker &> /dev/null; then
        print_success "Docker $(docker --version | cut -d' ' -f3) found"
    else
        print_warning "Docker not found (required for production mode)"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python $(python3 --version | cut -d' ' -f2) found"
    else
        print_warning "Python 3 not found (required for device client)"
    fi
    
    # Check Flutter (optional)
    if command -v flutter &> /dev/null; then
        print_success "Flutter found"
    else
        print_warning "Flutter not found (required for mobile app)"
    fi
    
    if [ $missing -eq 1 ]; then
        print_error "Missing required dependencies. Please install them first."
        exit 1
    fi
}

# Install dependencies
install_deps() {
    print_step "Installing Dependencies"
    
    if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "server" ]; then
        print_info "Installing server dependencies..."
        cd distributed_server
        npm install
        cd ..
        print_success "Server dependencies installed"
    fi
    
    if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "dashboard" ]; then
        print_info "Installing dashboard dependencies..."
        cd web_dashboard
        npm install
        cd ..
        print_success "Dashboard dependencies installed"
    fi
    
    if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "client" ]; then
        print_info "Installing Python client dependencies..."
        cd device_client
        pip3 install -r requirements.txt --quiet
        cd ..
        print_success "Python client dependencies installed"
    fi
}

# Start development mode
start_dev() {
    print_step "Starting Development Mode"
    
    # Create logs directory
    mkdir -p logs
    
    if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "server" ]; then
        print_info "Starting server on port 3001..."
        cd distributed_server
        npm run dev > ../logs/server.log 2>&1 &
        SERVER_PID=$!
        echo $SERVER_PID > ../logs/server.pid
        cd ..
        print_success "Server started (PID: $SERVER_PID)"
    fi
    
    sleep 2
    
    if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "dashboard" ]; then
        print_info "Starting dashboard on port 3000..."
        cd web_dashboard
        npm run dev > ../logs/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        echo $DASHBOARD_PID > ../logs/dashboard.pid
        cd ..
        print_success "Dashboard started (PID: $DASHBOARD_PID)"
    fi
    
    sleep 2
    
    echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║               🎉 Services Started Successfully!           ║${NC}"
    echo -e "${GREEN}╠═══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║   📊 Dashboard:  ${CYAN}http://localhost:3000${GREEN}                  ║${NC}"
    echo -e "${GREEN}║   🖥️  Server:     ${CYAN}http://localhost:3001${GREEN}                  ║${NC}"
    echo -e "${GREEN}║   📋 API Docs:   ${CYAN}http://localhost:3001/health${GREEN}           ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${YELLOW}To connect a device client:${NC}"
    echo -e "  ${CYAN}cd device_client && python3 src/client.py${NC}"
    
    echo -e "\n${YELLOW}To stop all services:${NC}"
    echo -e "  ${CYAN}./scripts/start.sh stop${NC}"
    
    # Wait for user interrupt
    echo -e "\n${PURPLE}Press Ctrl+C to stop all services...${NC}\n"
    
    # Trap Ctrl+C
    trap 'stop_services' INT
    
    # Tail logs
    tail -f logs/server.log logs/dashboard.log 2>/dev/null || wait
}

# Start production mode with Docker
start_production() {
    print_step "Starting Production Mode with Docker"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required for production mode"
        exit 1
    fi
    
    print_info "Building and starting containers..."
    docker-compose -f docker-compose.distributed.yml up --build -d
    
    print_success "All containers started!"
    
    echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           🚀 Production Services Running!                 ║${NC}"
    echo -e "${GREEN}╠═══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║   📊 Dashboard:  ${CYAN}http://localhost:3000${GREEN}                  ║${NC}"
    echo -e "${GREEN}║   🖥️  Server:     ${CYAN}http://localhost:3001${GREEN}                  ║${NC}"
    echo -e "${GREEN}║   🗄️  MongoDB:    ${CYAN}localhost:27017${GREEN}                       ║${NC}"
    echo -e "${GREEN}║   💾 Redis:      ${CYAN}localhost:6379${GREEN}                        ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${YELLOW}View logs:${NC}"
    echo -e "  ${CYAN}docker-compose -f docker-compose.distributed.yml logs -f${NC}"
    
    echo -e "\n${YELLOW}Stop services:${NC}"
    echo -e "  ${CYAN}./scripts/start.sh stop${NC}"
}

# Stop services
stop_services() {
    print_step "Stopping Services"
    
    # Stop development processes
    if [ -f logs/server.pid ]; then
        SERVER_PID=$(cat logs/server.pid)
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill $SERVER_PID
            print_success "Server stopped"
        fi
        rm logs/server.pid
    fi
    
    if [ -f logs/dashboard.pid ]; then
        DASHBOARD_PID=$(cat logs/dashboard.pid)
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
            kill $DASHBOARD_PID
            print_success "Dashboard stopped"
        fi
        rm logs/dashboard.pid
    fi
    
    # Stop Docker containers
    if command -v docker &> /dev/null; then
        docker-compose -f docker-compose.distributed.yml down 2>/dev/null || true
        print_success "Docker containers stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Show help
show_help() {
    echo -e "${CYAN}Usage: ./scripts/start.sh [mode] [component]${NC}"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo "  dev         Start in development mode (default)"
    echo "  prod        Start in production mode with Docker"
    echo "  install     Install dependencies only"
    echo "  stop        Stop all services"
    echo "  help        Show this help message"
    echo ""
    echo -e "${YELLOW}Components:${NC}"
    echo "  all         Start all services (default)"
    echo "  server      Start only the server"
    echo "  dashboard   Start only the dashboard"
    echo "  client      Start only the Python client"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./scripts/start.sh              # Start all in dev mode"
    echo "  ./scripts/start.sh prod         # Start all in production"
    echo "  ./scripts/start.sh dev server   # Start only server in dev"
    echo "  ./scripts/start.sh install      # Install all dependencies"
}

# Main
case $MODE in
    "dev")
        check_prerequisites
        install_deps
        start_dev
        ;;
    "prod")
        check_prerequisites
        start_production
        ;;
    "install")
        check_prerequisites
        install_deps
        print_success "All dependencies installed!"
        ;;
    "stop")
        stop_services
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown mode: $MODE"
        show_help
        exit 1
        ;;
esac
