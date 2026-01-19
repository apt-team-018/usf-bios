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

# Start backend API (internal only - bound to 127.0.0.1)
echo "Starting backend..."
cd /app/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
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
echo "  Access: http://localhost:3000"
echo ""
echo "=============================================="

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
