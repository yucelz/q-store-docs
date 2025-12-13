---
title: Kubernetes Deployment
description: Deploying Q-Store on Kubernetes
---

Production-ready Kubernetes deployment for Q-Store.

## Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-db-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-db
  template:
    spec:
      containers:
      - name: api
        image: quantum-db:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: quantum-db-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: quantum-db
```

## Next Steps

- See [Cloud Deployment](/deployment/cloud) for detailed setup
- Check [Migration Guide](/deployment/migration)
