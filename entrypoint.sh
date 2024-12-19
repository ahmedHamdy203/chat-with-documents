#!/bin/bash

# Start the FastAPI backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend &

# Start a simple HTTP server for the frontend
cd frontend && python -m http.server 8080 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?