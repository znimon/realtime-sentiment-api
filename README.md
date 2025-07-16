# Real-time Sentiment Analysis API

![Architecture Diagram](docs/assets/images/architecture.png)

Real-time API for sentiment analysis, built with FastAPI and Redis.

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Performance](#performance)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities
- Real-time sentiment prediction (POSITIVE/NEGATIVE/NEUTRAL)
- Batch processing for high-throughput scenarios
- Automatic model caching and warm-up

### Infrastructure
- Docker containerization
- Redis caching layer
- Prometheus + Grafana monitoring
- Health checks and readiness probes

## Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.4+

### Installation
1. Clone the repository:
git clone https://github.com/znimon/sentiment-analysis-api.git
cd sentiment-analysis-api

2. Start all services:
docker compose up -d --build

3. Verify installation:
curl http://localhost:8000/health

## API Documentation

### Endpoint Summary

| Endpoint          | Method | Description                     |
|-------------------|--------|---------------------------------|
| /predict         | POST   | Single text prediction          |
| /batch_predict   | POST   | Batch predictions               |
| /health          | GET    | System health status            |
| /metrics         | GET    | Prometheus metrics              |
| /docs            | GET    | Interactive API documentation   |

### Sample Request
POST /predict
Content-Type: application/json

{
  "text": "I really enjoy using this product"
}

### Sample Response
{
  "text": "I really enjoy using this product",
  "sentiment": "POSITIVE",
  "confidence": 0.92,
  "processing_time_ms": 45,
  "cached": false
}

## Configuration

### Environment Variables
| Variable        | Default Value                                      | Description                |
|-----------------|----------------------------------------------------|----------------------------|
| REDIS_URL       | redis://redis:6379                                 | Redis connection URL       |
| MODEL_NAME      | cardiffnlp/twitter-roberta-base-sentiment-latest   | HuggingFace model name     |
| BATCH_SIZE      | 32                                                 | Batch processing size      |
| CACHE_TTL       | 3600                                               | Cache duration in seconds  |

## Monitoring

### Included Dashboards
1. API Performance Overview
   - Request rates
   - Error rates
   - Latency percentiles

2. System Health
   - CPU/Memory usage
   - Service status
   - Uptime monitoring

## Performance

### Benchmark Results
| Test Scenario       | Throughput | P95 Latency | Cache Hit Rate |
|---------------------|------------|-------------|----------------|
| Single (cached)     | 1200 req/s | 15ms        | 100%           |
| Single (uncached)   | 200 req/s  | 450ms       | 0%             |
| Batch (10 items)    | 80 req/s   | 600ms       | 40%            |

## Development

### Project Structure
src/
  api/               # FastAPI application code
  model/             # ML model implementation
  services/          # Business logic
  monitoring/       # Metrics collection

### Rebuild

docker-compose down && docker-compose build --no-cache sentiment-api && docker-compose up -d

### Running Tests
pytest tests/unit/       # Unit tests  
pytest tests/integration # Integration tests  

## Deployment

### Production Checklist
- Configure TLS termination
- Set resource limits in Docker
- Enable log rotation
- Configure alert thresholds
- Set up backup for Redis data

### Kubernetes Example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: your-registry/sentiment-api:1.0.0
        ports:
        - containerPort: 8000

## Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request

## License
MIT License
Copyright (c) 2024 Zechariah Nimon