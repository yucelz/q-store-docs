---
title: Cloud Deployment
description: Deploying Q-Store to cloud platforms
---

Deploy Q-Store on major cloud platforms with production-ready configurations.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│          Load Balancer (AWS ALB)        │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────┐              ┌────▼───┐
│  API   │              │  API   │
│ Server │              │ Server │
│  Pod   │              │  Pod   │
└───┬────┘              └────┬───┘
    │                        │
    └────────────┬───────────┘
                 │
    ┌────────────▼────────────┐
    │                         │
┌───▼──────┐          ┌──────▼────┐
│ Pinecone │          │   IonQ    │
│  Cloud   │          │  Quantum  │
└──────────┘          └───────────┘
```

## AWS Deployment

### Infrastructure as Code (Terraform)

```hcl
# main.tf

# VPC Configuration
resource "aws_vpc" "quantum_db" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  
  tags = {
    Name = "quantum-db-vpc"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "quantum_db" {
  name = "quantum-db-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "quantum_db_api" {
  family                   = "quantum-db-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"
  
  container_definitions = jsonencode([{
    name  = "api"
    image = "your-repo/quantum-db:latest"
    
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    
    environment = [
      {
        name  = "PINECONE_ENVIRONMENT"
        value = "us-east-1"
      }
    ]
    
    secrets = [
      {
        name      = "IONQ_API_KEY"
        valueFrom = aws_secretsmanager_secret.ionq_key.arn
      },
      {
        name      = "PINECONE_API_KEY"
        valueFrom = aws_secretsmanager_secret.pinecone_key.arn
      }
    ]
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/quantum-db"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

# ECS Service
resource "aws_ecs_service" "quantum_db_api" {
  name            = "quantum-db-api"
  cluster         = aws_ecs_cluster.quantum_db.id
  task_definition = aws_ecs_task_definition.quantum_db_api.arn
  desired_count   = 3
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# Application Load Balancer
resource "aws_lb" "quantum_db" {
  name               = "quantum-db-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

# Secrets Manager
resource "aws_secretsmanager_secret" "ionq_key" {
  name = "quantum-db/ionq-api-key"
}

resource "aws_secretsmanager_secret" "pinecone_key" {
  name = "quantum-db/pinecone-api-key"
}
```

### Deploy with AWS CLI

```bash
# Build and push Docker image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

docker build -t quantum-db .
docker tag quantum-db:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/quantum-db:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/quantum-db:latest

# Deploy with Terraform
terraform init
terraform plan
terraform apply
```

## GCP Deployment

### Google Kubernetes Engine

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-db-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-db
  template:
    metadata:
      labels:
        app: quantum-db
    spec:
      containers:
      - name: api
        image: gcr.io/your-project/quantum-db:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: PINECONE_ENVIRONMENT
          value: "us-east-1"
        - name: IONQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: quantum-db-secrets
              key: ionq-api-key
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: quantum-db-secrets
              key: pinecone-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: quantum-db-api
  namespace: production
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: quantum-db

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-db-api
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-db-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to GKE

```bash
# Build and push image
gcloud builds submit --tag gcr.io/your-project/quantum-db:latest

# Create cluster
gcloud container clusters create quantum-db-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --region=us-central1

# Create secrets
kubectl create secret generic quantum-db-secrets \
  --from-literal=ionq-api-key=$IONQ_API_KEY \
  --from-literal=pinecone-api-key=$PINECONE_API_KEY \
  --namespace=production

# Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

## Azure Deployment

### Azure Container Instances

```yaml
# azure-deployment.yaml
apiVersion: '2021-09-01'
location: eastus
name: quantum-db-api
properties:
  containers:
  - name: api
    properties:
      image: your-registry.azurecr.io/quantum-db:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: PINECONE_ENVIRONMENT
        value: us-east-1
      - name: IONQ_API_KEY
        secureValue: your-ionq-key
      - name: PINECONE_API_KEY
        secureValue: your-pinecone-key
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 80
    dnsNameLabel: quantum-db-api
type: Microsoft.ContainerInstance/containerGroups
```

### Deploy to Azure

```bash
# Build and push
az acr build --registry yourregistry \
  --image quantum-db:latest .

# Deploy
az container create \
  --resource-group quantum-db-rg \
  --file azure-deployment.yaml
```

## Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - IONQ_API_KEY=${IONQ_API_KEY}
      - PINECONE_ENVIRONMENT=us-east-1
    depends_on:
      - redis
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

## Environment Variables

### Required

```bash
IONQ_API_KEY=your-ionq-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1
```

### Optional

```bash
# Quantum settings
QUANTUM_TARGET_DEVICE=simulator  # or qpu.aria, qpu.forte
MAX_CONCURRENT_CIRCUITS=10
CIRCUIT_CACHE_SIZE=1000

# Connection pooling
MAX_CONNECTIONS=100
MIN_CONNECTIONS=10
CONNECTION_TIMEOUT=30

# Performance
ENABLE_CACHE=true
CACHE_TTL=300
LOG_LEVEL=INFO
```

## Health Checks

### Liveness Probe

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Readiness Probe

```python
@app.get("/ready")
async def readiness_check():
    # Check classical backend
    classical_ok = await db.classical_backend.ping()
    
    # Check quantum backend
    quantum_ok = await db.quantum_backend.ping()
    
    if classical_ok and quantum_ok:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Not ready")
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# Auto-scaling based on metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: quantum_query_latency_p95
      target:
        type: AverageValue
        averageValue: "1000"  # 1 second
```

### Vertical Scaling

```yaml
# Resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
```

## Monitoring

See [Monitoring Guide](/production/monitoring) for complete setup.

## Security

- Store API keys in secrets manager
- Use IAM roles for cloud resources
- Enable TLS/SSL for all connections
- Network isolation with VPC
- Regular security audits

## Next Steps

- Set up [Kubernetes Deployment](/deployment/kubernetes)
- Configure [Migration](/deployment/migration)
- Implement [Monitoring](/production/monitoring)
