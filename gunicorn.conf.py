import os

timeout = 600
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