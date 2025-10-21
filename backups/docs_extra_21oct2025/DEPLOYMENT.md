# Hướng dẫn Deployment - Askly

## Tổng quan

Tài liệu này hướng dẫn deploy Askly lên các môi trường khác nhau: Development, Staging, và Production.

## 📋 Checklist trước khi Deploy

- [ ] Đã test kỹ trên local
- [ ] Có API keys (Gemini)
- [ ] Đã build embeddings
- [ ] Cấu hình environment variables
- [ ] Backup dữ liệu quan trọng
- [ ] Setup monitoring và logging

## 🐳 Docker Deployment

### 1. Tạo Dockerfile

**Backend Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download vi_core_news_lg

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "api_server.py"]
```

**Frontend Dockerfile**:
```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY streamlit_app/front-end/package*.json ./
RUN npm install

# Copy source
COPY streamlit_app/front-end/ .

# Build
RUN npm run build

# Install serve
RUN npm install -g serve

# Expose port
EXPOSE 5173

# Serve build
CMD ["serve", "-s", "dist", "-l", "5173"]
```

### 2. Docker Compose

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - API_HOST=0.0.0.0
      - API_PORT=5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  data:
  models:
```

### 3. Chạy với Docker

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☁️ Cloud Deployment

### AWS EC2

#### 1. Chuẩn bị EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 22.04, t3.medium hoặc lớn hơn)
# Security Group: Mở port 22, 5000, 5173

# SSH vào instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. Deploy Application

```bash
# Clone repository
git clone <your-repo>
cd askly

# Setup environment
cp .env.example .env
nano .env  # Thêm API keys

# Build embeddings (nếu chưa có)
python run.py --build

# Start with Docker
docker-compose up -d

# Setup nginx reverse proxy (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/askly
```

**Nginx config**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/askly /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Google Cloud Platform (GCP)

#### 1. Cloud Run Deployment

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/askly-backend

# Deploy to Cloud Run
gcloud run deploy askly-backend \
  --image gcr.io/PROJECT_ID/askly-backend \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key \
  --memory 4Gi \
  --cpu 2
```

#### 2. App Engine Deployment

**app.yaml**:
```yaml
runtime: python310
instance_class: F4

env_variables:
  GEMINI_API_KEY: "your_key"

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.65

handlers:
- url: /.*
  script: auto
```

```bash
# Deploy
gcloud app deploy
```

### Heroku

#### 1. Chuẩn bị

**Procfile**:
```
web: python api_server.py
```

**runtime.txt**:
```
python-3.10.12
```

#### 2. Deploy

```bash
# Login
heroku login

# Create app
heroku create askly-app

# Set environment variables
heroku config:set GEMINI_API_KEY=your_key

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1

# View logs
heroku logs --tail
```

### DigitalOcean

#### 1. App Platform

```yaml
# .do/app.yaml
name: askly
services:
- name: backend
  github:
    repo: your-username/askly
    branch: main
  build_command: pip install -r requirements.txt
  run_command: python api_server.py
  envs:
  - key: GEMINI_API_KEY
    value: ${GEMINI_API_KEY}
  instance_count: 1
  instance_size_slug: professional-xs
  http_port: 5000
  
- name: frontend
  github:
    repo: your-username/askly
    branch: main
  build_command: cd streamlit_app/front-end && npm install && npm run build
  run_command: cd streamlit_app/front-end && npm run preview
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 5173
```

## 🔧 Production Configuration

### 1. Environment Variables

```bash
# Production .env
GEMINI_API_KEY=your_production_key
API_HOST=0.0.0.0
API_PORT=5000
ENVIRONMENT=production

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/askly/app.log

# Performance
MAX_WORKERS=4
TIMEOUT=30
```

### 2. Gunicorn (Production WSGI Server)

**requirements.txt** thêm:
```
gunicorn==21.2.0
```

**gunicorn_config.py**:
```python
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '/var/log/askly/access.log'
errorlog = '/var/log/askly/error.log'
loglevel = 'info'

# Process naming
proc_name = 'askly'

# Server mechanics
daemon = False
pidfile = '/var/run/askly.pid'
user = None
group = None
tmp_upload_dir = None
```

**Chạy với Gunicorn**:
```bash
gunicorn -c gunicorn_config.py api_server:app
```

### 3. Systemd Service

**/etc/systemd/system/askly.service**:
```ini
[Unit]
Description=Askly RAG System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/askly
Environment="PATH=/home/ubuntu/askly/venv/bin"
ExecStart=/home/ubuntu/askly/venv/bin/gunicorn -c gunicorn_config.py api_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable askly
sudo systemctl start askly
sudo systemctl status askly

# View logs
sudo journalctl -u askly -f
```

## 📊 Monitoring & Logging

### 1. Application Logging

**logging_config.py**:
```python
import logging
import logging.handlers
import os

def setup_logging():
    log_dir = os.getenv('LOG_DIR', '/var/log/askly')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('askly')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 2. Prometheus Metrics

```bash
pip install prometheus-flask-exporter
```

```python
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Custom metrics
request_duration = metrics.histogram(
    'request_duration_seconds',
    'Request duration',
    labels={'endpoint': lambda: request.endpoint}
)
```

### 3. Health Checks

```python
@app.route('/health')
def health_check():
    checks = {
        'status': 'healthy',
        'pipeline': pipeline is not None,
        'embeddings': check_embeddings_loaded(),
        'llm': check_llm_available(),
        'disk_space': check_disk_space()
    }
    
    status_code = 200 if all(checks.values()) else 503
    return jsonify(checks), status_code
```

## 🔒 Security Best Practices

### 1. API Security

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")
def ask_question():
    # ...
```

### 2. HTTPS/SSL

```bash
# Let's Encrypt SSL
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 3. Environment Variables

```bash
# Không hardcode secrets
# Sử dụng secrets management
# AWS: AWS Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault
```

## 🔄 CI/CD Pipeline

### GitHub Actions

**.github/workflows/deploy.yml**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd /home/ubuntu/askly
          git pull origin main
          docker-compose down
          docker-compose up -d --build
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3
    # ...
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - backend
```

**nginx.conf** (Load Balancer):
```nginx
upstream backend {
    least_conn;
    server backend_1:5000;
    server backend_2:5000;
    server backend_3:5000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://backend;
    }
}
```

### Vertical Scaling

```yaml
# Tăng resources
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Tăng swap
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Port already in use**
   ```bash
   sudo lsof -i :5000
   sudo kill -9 <PID>
   ```

3. **Permission denied**
   ```bash
   sudo chown -R $USER:$USER /path/to/askly
   chmod +x start_backend.sh
   ```

## 📝 Checklist sau khi Deploy

- [ ] Health check endpoint hoạt động
- [ ] API endpoints trả về đúng
- [ ] Frontend kết nối được backend
- [ ] Logs được ghi đúng
- [ ] Monitoring dashboard hoạt động
- [ ] SSL certificate valid
- [ ] Backup được setup
- [ ] Alerts được cấu hình

---

**Lưu ý**: Luôn test kỹ trên staging trước khi deploy lên production!
