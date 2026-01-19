#!/bin/bash
set -e

echo "=============================================="
echo "  USF BIOS Web UI"
echo "  Powered by US Inc"
echo "=============================================="
echo ""

# Verify backend files exist
echo "Verifying backend files..."
if [ ! -f "/app/backend/app/models/db_models.py" ]; then
    echo "ERROR: db_models.py not found!"
    ls -la /app/backend/app/models/
    exit 1
fi
echo "  ✓ Backend files verified"

# Start backend API
echo "Starting backend..."
cd /app/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ Backend is ready"
        break
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend..."
cd /app/frontend
node server.js &
FRONTEND_PID=$!

# Wait for frontend to be ready
sleep 3

echo ""
echo "=============================================="
echo "  Services Started Successfully!"
echo "=============================================="
echo ""
echo "  Frontend UI:  http://localhost:3000"
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo ""
echo "=============================================="

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
