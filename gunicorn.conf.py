import os

# Increase timeout for long-running requests
timeout = 1800  # 30 minutes
workers = 1
preload_app = True
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
keepalive = 120
worker_class = 'sync'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Graceful timeout
graceful_timeout = 300

# Worker timeout
worker_timeout = 1800  # 30 minutes