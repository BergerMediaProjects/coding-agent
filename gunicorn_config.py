import multiprocessing

# Gunicorn configuration for production
bind = "127.0.0.1:8000"  # Only listen locally, Nginx will proxy
workers = multiprocessing.cpu_count() * 2 + 1  # Number of worker processes
worker_class = "sync"  # Use sync workers
timeout = 3600  # Set timeout to 60 minutes for long-running pipelines
keepalive = 5  # Keepalive timeout
max_requests = 1000  # Restart workers after this many requests
max_requests_jitter = 50  # Add random jitter to max_requests
daemon = False  # Don't daemonize in production (let systemd handle it)
accesslog = "data/log/gunicorn_access.log"  # Access log location
errorlog = "data/log/gunicorn_error.log"  # Error log location
loglevel = "info"  # Log level 
