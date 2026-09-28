#!/bin/bash
# run.sh - Execution script for MAORS

echo "=========================================="
echo "    Starting MAORS Clinical AI Platform   "
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "[!] Docker could not be found. Please install Docker."
    exit 1
fi

# Build and run the containers
echo "[*] Building and starting Docker containers..."
docker-compose up --build -d

echo ""
echo "[*] MAORS is booting up! Please wait a few seconds."
echo "[*] The application will be available at: http://localhost:8080"
echo "[*] To stop the application, run: docker-compose down"
echo "=========================================="
