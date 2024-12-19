import subprocess
import webbrowser
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

def start_backend():
    """Start the FastAPI backend"""
    print("Starting backend server...")
    return subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="backend"
    )

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("Starting frontend server...")
    os.chdir("frontend")
    httpd = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
    return httpd

def main():
    # Start the backend
    backend_process = start_backend()
    
    # Start the frontend server
    frontend_server = start_frontend()
    
    try:
        # Wait a bit for servers to start
        time.sleep(2)
        
        # Open the browser
        print("Opening browser...")
        webbrowser.open('http://localhost:8080')
        
        # Keep the frontend server running
        frontend_server.serve_forever()
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_server.shutdown()
        print("Servers stopped")

if __name__ == "__main__":
    main()